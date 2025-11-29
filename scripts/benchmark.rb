require 'open3'
require 'benchmark'
require 'time'

# --- Configuration Constants ---

# Paths to the reference genomes and read files
REF_B38    = "/home/jkane/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"
REF_CHM13  = "/home/jkane/Genomics/Reference/chm13v2.0/chm13v2.0.fa.gz"
R1         = "/home/jkane/Genomics/HG002/2A1_CGATGT_L001_R1_001.fastq.gz"
R2         = "/home/jkane/Genomics/HG002/2A1_CGATGT_L001_R2_001.fastq.gz"

# Paths to the alignment tools
FERROUS_ALIGN_BIN = "/home/jkane/RustroverProjects/FerrousAlign/target/release/ferrous-align"

# Environment variable used to control the SIMD backend
SIMD_ENV_VAR = "FERROUS_ALIGN_FORCE_AVX2"

BWA        = "/home/jkane/Applications/bwa-mem2/bwa-mem2"
BWA_AVX2   = "/home/jkane/Applications/bwa-mem2/bwa-mem2.avx2"
# If available, you can also benchmark AVX-512 variant by adding:
# BWA_AVX512 = "/home/jkane/Applications/bwa-mem2/bwa-mem2.avx512bw"

# Common settings
THREADS    = 16
BATCH_SIZE = 500000
TOTAL_READ_PAIRS = 4_000_000 # 4M
TOTAL_READS = TOTAL_READ_PAIRS * 2 # 8M

# A simple class to hold the benchmark results for each run
BenchmarkResult = Struct.new(
  :name,
  :tool,
  :reference,
  :sam_path,
  :log_path,
  :wall_time_s,
  :max_rss_kb,
  :record_count,
  :exit_status,
  :simd_width,
  :simd_parallelism,
  :user_time_s,
  :system_time_s,
  :cpu_time_s,
  :cpu_efficiency_pct,
  :mapped_percent,
  :properly_paired_percent,
  :pairs_rescued
) do
  # Calculates the throughput in reads per second (reads/s)
  def throughput
    record_count / wall_time_s.to_f
  end

  # Formats wall time from seconds into the "M:SS.ms" format
  def format_wall_time
    minutes = (wall_time_s / 60).to_i
    seconds = wall_time_s % 60
    "#{minutes}:#{sprintf('%05.2f', seconds)}"
  end
end


# --- Helper Methods ---

# Run `samtools flagstats` on a SAM file and parse key percentages
def run_flagstats(sam_path)
  mapped_percent = nil
  properly_paired_percent = nil

  begin
    # Use Open3 to capture stdout, stderr
    stdout_str, stderr_str, status = Open3.capture3("samtools", "flagstats", sam_path)
    if status.success?
      stdout_str.each_line do |line|
        if line =~ /\bmapped\b.*\(([^%]+)%/
          mapped_percent = $1.to_f
        elsif line =~ /\bproperly paired\b.*\(([^%]+)%/
          properly_paired_percent = $1.to_f
        end
      end
    else
      warn "samtools flagstats failed for #{sam_path}: #{stderr_str}"
    end
  rescue StandardError => e
    warn "Error running samtools flagstats on #{sam_path}: #{e.message}"
  end

  { mapped_percent: mapped_percent, properly_paired_percent: properly_paired_percent }
end

# Helper function to get the detailed CPU model name
def get_cpu_model
  # Run lscpu and grep for 'Model name', then pipe to cut and xargs to clean it up.
  # We use the backtick operator (`) to execute the shell command.
  
  # Original shell command: lscpu | grep 'Model name' | cut -d: -f2 | xargs
  
  # A more robust Ruby way to execute a command and capture output:
  output = `lscpu | grep 'Model name:'`
  
  # Extract and clean the name using Ruby regex
  if output =~ /Model name:\s*(.*)/
    return $1.strip
  else
    return "x86_64 Processor (lscpu detail unavailable)"
  end
rescue StandardError
  return "x86_64 Processor (lscpu execution failed)"
end

# Executes a benchmark command and parses its output
def run_benchmark(name, old_tool_path, reference, sam_path, log_path, args)
  puts "=== #{name} ==="

  tool_name_only = old_tool_path.split('/').last
  env_vars = {} # Initialize environment variable hash
  
  # 1. Determine the actual binary path and necessary environment variables
  if tool_name_only.include?("ferrous-align")
    tool = FERROUS_ALIGN_BIN
    
    if name.include?("AVX2")
      # AVX2 run: Set FERROUS_ALIGN_FORCE_AVX2=1
      env_vars[SIMD_ENV_VAR] = "1"
      tool_label = "ferrous-align-avx2" # For log output consistency
    else # AVX-512 run
      # AVX-512 run: Set FERROUS_ALIGN_FORCE_AVX2=0
      env_vars[SIMD_ENV_VAR] = "0"
      tool_label = "ferrous-align-avx512" # For log output consistency
    end
  else
    tool = old_tool_path # BWA-MEM2 uses the original path
    tool_label = tool_name_only
  end

  # 2. Construct the command (executable and array of arguments)
  command_args = ["/usr/bin/time", "-v", tool, "mem", "-t", THREADS.to_s, *args, reference, R1, R2]
  
  cmd_str = command_args.join(' ')
  puts "Running: #{env_vars.empty? ? '' : env_vars.map { |k, v| "#{k}=#{v}" }.join(' ')} #{cmd_str} > #{sam_path} 2> #{log_path}"

wall_time = 0.0
  max_rss = 0
  simd_width = "N/A"
  simd_parallelism = "N/A"
  pairs_rescued = nil
  exit_status = nil
  user_time_s = nil
  system_time_s = nil

# Use Benchmark.measure to get the accurate wall clock time.
  timing_result = Benchmark.measure do
    # Run command using Open3.popen3. Pass the environment variables as the first argument (a hash).
    # If env_vars is empty (like for BWA-MEM2), just pass the command arguments.
    # Note: Open3 accepts the environment hash OR the command as its first arg.
    
# CORRECT WAY TO CALL Open3.popen3 with and without an Environment Hash:
    if env_vars.empty?
      # For BWA-MEM2 (no special env vars)
      Open3.popen3(*command_args) do |stdin, stdout, stderr, wait_thr|
        # Write SAM output to file
        File.open(sam_path, 'w') do |f|
          while line = stdout.gets
            f.puts(line)
          end
        end
        # Write /usr/bin/time -v output AND tool-specific output to log file
        File.open(log_path, 'w') do |f|
          while line = stderr.gets
            f.puts(line)
          end
        end
        exit_status = wait_thr.value
      end
    else
      # For ferrous-align (with FERROUS_ALIGN_FORCE_AVX2 set)
      # Pass the environment hash as the first argument, followed by the command arguments.
      Open3.popen3(env_vars, *command_args) do |stdin, stdout, stderr, wait_thr|
        
        # Write SAM output to file
        File.open(sam_path, 'w') do |f|
          while line = stdout.gets
            f.puts(line)
          end
        end
        # Write /usr/bin/time -v output AND tool-specific output to log file
        File.open(log_path, 'w') do |f|
          while line = stderr.gets
            f.puts(line)
          end
        end
        exit_status = wait_thr.value
      end
    end
  end

  # Extract the wall clock time (real time) from the measure result object
  wall_time_s = timing_result.real

  # 3. Parse the log file for timing, memory, and SIMD info
  if File.exist?(log_path)
    File.readlines(log_path).each do |line|
      # Parse /usr/bin/time -v output
      if line =~ /Maximum resident set size \(kbytes\):\s*(\d+)/
        max_rss = $1.to_i
      end
      if line =~ /User time \(seconds\):\s*([0-9\.]+)/
        user_time_s = $1.to_f
      end
      if line =~ /System time \(seconds\):\s*([0-9\.]+)/
        system_time_s = $1.to_f
      end

      # Parse ferrous-align SIMD info
      # Examples: 
      #  - [INFO ] Using compute backend: AVX-512 (512-bit, 64-way parallelism)
      #  - [INFO ] Using compute backend: AVX2 (256-bit, 32-way parallelism)
      if line =~ /Using compute backend:\s*([A-Z0-9\-]+)\s*\((\d+)-bit,\s*(\d+)-way parallelism\)/
        backend = $1
        bits = $2
        ways = $3
        simd_width = "#{bits}-bit (#{backend})"
        simd_parallelism = ways
      end

      # Parse ferrous-align pairs rescued from completion line
      # Example: [INFO ] Complete: ..., 883888 pairs rescued in 2101.415 CPU sec, 163.657 real sec
      if line =~ /pairs rescued/i
        if line =~ /([0-9,]+)\s+pairs rescued/i
          pairs_rescued = $1.delete(',').to_i
        end
      end

      # Parse BWA-MEM2 SIMD info from its log line
      # Example: Looking to launch executable ".*/bwa-mem2.avx512bw", simd = .avx512bw
      if line =~ /simd\s*=\s*\.([a-z0-9_]+)/i
        simd_token = $1.downcase
        if simd_token.start_with?("avx512")
          simd_width = "512-bit (#{simd_token.upcase})"
        elsif simd_token.include?("avx2")
          simd_width = "256-bit (AVX2)"
        elsif simd_token.start_with?("sse")
          simd_width = "128-bit (#{simd_token.upcase})"
        else
          simd_width = simd_token.upcase
        end
        # BWA-MEM2 doesn't report per-lane parallelism like ferrous-align
        simd_parallelism = "1"
      end
    end
  end

  # 4. Count records in the SAM file
  record_count = `grep -v '^@' #{sam_path} | wc -l`.strip.to_i

  # 5. Compute CPU aggregates
  cpu_time_s = nil
  cpu_efficiency_pct = nil
  if user_time_s && system_time_s
    cpu_time_s = user_time_s + system_time_s
    cpu_efficiency_pct = (cpu_time_s / (timing_result.real.nonzero? || 1.0)) * 100.0
  end

  # 6. Run samtools flagstats for mapped and properly paired percentages
  fs = run_flagstats(sam_path)
  mapped_percent = fs[:mapped_percent]
  properly_paired_percent = fs[:properly_paired_percent]

puts "Exit: #{exit_status.exitstatus}"
  puts "Elapsed (Benchmark.real): #{'%.2f' % wall_time_s}s"
  puts "Max RSS: #{max_rss} KB"
  puts "SIMD: #{simd_width} (#{simd_parallelism}-way)"
  puts ""

  BenchmarkResult.new(
    name, tool_label, reference.split('/').last, sam_path, log_path,
    wall_time_s, max_rss, record_count, exit_status,
    simd_width, simd_parallelism.to_i,
    user_time_s, system_time_s, cpu_time_s, cpu_efficiency_pct,
    mapped_percent, properly_paired_percent,
    pairs_rescued
  )
end

# Prints the benchmark summary tables
def print_summary_tables(results)
  puts "========================================"
  puts "Summary Tables"
  puts "========================================"
  puts ""

  # --- Table 1: Time and Memory by Tool/Reference ---
  puts "### ⏱️ Alignment Performance"
  puts "| Tool | Reference | SIMD | Wall Time | Max Memory (GB) | Throughput (reads/s) |"
  puts "|:---|:---|:---:|:---:|:---:|:---:|"

  results.each do |r|
    tool_name = r.tool.start_with?('ferrous') ? r.tool.split('-').last.upcase : 'BWA-MEM2'
    reference_name = r.reference.include?('GRCh38') ? 'GRCh38' : 'CHM13v2.0'
    max_memory_gb = '%.2f' % (r.max_rss_kb / 1024.0 / 1024.0)

    simd_label = r.simd_width.split(' ').first # e.g., 128-bit
    
    puts "| **#{tool_name}** | #{reference_name} | #{simd_label} | #{r.format_wall_time} | #{max_memory_gb} | #{r.throughput.to_i.to_s.reverse.gsub(/(\d{3})(?=\d)/, '\\1,').reverse} |"
  end
  puts ""
  puts "---"

  # --- Table 2: Detailed Results (for the first run) ---
  puts "### 🔬 Detailed Run Metrics (Example: #{results.first.name})"
  first_run = results.first
  total_records = first_run.record_count

  # Calculate supplementary records
  supplementary = total_records - TOTAL_READS
  
  # Mapped and properly paired from samtools flagstats (parsed earlier)
  mapped_percent = first_run.mapped_percent
  properly_paired_percent = first_run.properly_paired_percent
  
  # CPU time and efficiency from /usr/bin/time -v (parsed earlier)
  cpu_time = first_run.cpu_time_s
  cpu_efficiency = first_run.cpu_efficiency_pct
  
  simd_width_display = first_run.simd_width.start_with?("128") ? "128-bit (baseline)" : first_run.simd_width
  simd_parallelism_display = first_run.simd_parallelism == 0 ? "N/A" : "#{first_run.simd_parallelism}-way"


  puts "| Metric | Value |"
  puts "|:---|:---|"
  puts "| **SIMD Width** | **#{simd_width_display}** |"
  puts "| **Parallelism** | **#{simd_parallelism_display}** |"
  puts "| Total reads | #{TOTAL_READS.to_s.reverse.gsub(/(\d{3})(?=\d)/, '\\1,').reverse} (#{TOTAL_READ_PAIRS.to_s.reverse.gsub(/(\d{3})(?=\d)/, '\\1,').reverse} pairs) |"
  puts "| Total records | #{total_records.to_s.reverse.gsub(/(\d{3})(?=\d)/, '\\1,').reverse} |"
  puts "| Supplementary | #{supplementary.to_s.reverse.gsub(/(\d{3})(?=\d)/, '\\1,').reverse} |"
  if mapped_percent
    puts "| Mapped | #{sprintf('%.2f', mapped_percent)}% |"
  else
    puts "| Mapped | N/A |"
  end
  if properly_paired_percent
    puts "| Properly paired | #{sprintf('%.2f', properly_paired_percent)}% |"
  else
    puts "| Properly paired | N/A |"
  end
  puts "| **Wall time** | **#{first_run.format_wall_time} (#{sprintf('%.2f', first_run.wall_time_s)}s)** |"
  if cpu_time
    puts "| CPU time | #{sprintf('%.2f', cpu_time)}s |"
  else
    puts "| CPU time | N/A |"
  end
  puts "| **Throughput** | **#{first_run.throughput.to_i.to_s.reverse.gsub(/(\d{3})(?=\d)/, '\\1,').reverse} reads/sec** |"
  if cpu_efficiency
    puts "| CPU efficiency | #{sprintf('%.0f', cpu_efficiency)}% |"
  else
    puts "| CPU efficiency | N/A |"
  end
  if first_run.pairs_rescued
    puts "| Pairs rescued | #{first_run.pairs_rescued.to_s.reverse.gsub(/(\d{3})(?=\d)/, '\\1,').reverse} |"
  else
    puts "| Pairs rescued | N/A |"
  end
  puts "| Max memory | #{'%.2f' % (first_run.max_rss_kb / 1024.0 / 1024.0)} GB |"
end


# --- Main Execution Block ---

begin
  puts "========================================"
  puts "Benchmark: x86_64 AVX2 vs AVX-512"
  puts "Date: #{Time.now.strftime('%Y-%m-%d %H:%M:%S')}"
  # Note: Getting CPU model in Ruby is platform-dependent; a simple placeholder is used.
  puts "CPU: #{get_cpu_model}"
  puts "Threads: #{THREADS}"
  puts "Dataset: #{TOTAL_READ_PAIRS / 1_000_000}M HG002 read pairs (#{TOTAL_READS / 1_000_000}M reads)"
  puts "========================================"
  puts ""

results = []
  temp_dir = '/home/jkane/benches'

  # 1. AVX2 on GRCh38
  results << run_benchmark(
    "1. AVX2 on GRCh38", FERROUS_ALIGN_BIN, REF_B38, # Use FERROUS_ALIGN_BIN path
    "#{temp_dir}/avx2_grch38.sam", "#{temp_dir}/avx2_grch38.log",
    ["--batch-size", BATCH_SIZE.to_s]
  )

  # 2. AVX2 on CHM13v2.0
  results << run_benchmark(
    "2. AVX2 on CHM13v2.0", FERROUS_ALIGN_BIN, REF_CHM13, # Use FERROUS_ALIGN_BIN path
    "#{temp_dir}/avx2_chm13.sam", "#{temp_dir}/avx2_chm13.log",
    ["--batch-size", BATCH_SIZE.to_s]
  )

  # 3. AVX-512 on GRCh38
  results << run_benchmark(
    "3. AVX-512 on GRCh38", FERROUS_ALIGN_BIN, REF_B38, # Use FERROUS_ALIGN_BIN path
    "#{temp_dir}/avx512_grch38.sam", "#{temp_dir}/avx512_grch38.log",
    ["--batch-size", BATCH_SIZE.to_s]
  )

  # 4. AVX-512 on CHM13v2.0
  results << run_benchmark(
    "4. AVX-512 on CHM13v2.0", FERROUS_ALIGN_BIN, REF_CHM13, # Use FERROUS_ALIGN_BIN path
    "#{temp_dir}/avx512_chm13.sam", "#{temp_dir}/avx512_chm13.log",
    ["--batch-size", BATCH_SIZE.to_s]
  )

  # 5. BWA-MEM2 AVX2 on GRCh38 (reference) 
  results << run_benchmark(
    "5. BWA-MEM2 AVX2 on GRCh38 (reference)", BWA_AVX2, REF_B38,
    "#{temp_dir}/bwamem2_grch38.sam", "#{temp_dir}/bwamem2_grch38.log",
    [] 
  )

  # 6. BWA-MEM2 AVX2 on CHM13v2.0 (reference) 
  results << run_benchmark(
    "6. BWA-MEM2 AVX2 on CHM13v2.0 (reference)", BWA_AVX2, REF_CHM13,
    "#{temp_dir}/bwamem2_chm13.sam", "#{temp_dir}/bwamem2_chm13.log",
    []
  )

  # --- Print Summary ---
  print_summary_tables(results)

  puts "========================================"
  puts "Benchmark complete: #{Time.now.strftime('%Y-%m-%d %H:%M:%S')}"
  puts "========================================"

rescue StandardError => e
  STDERR.puts "An error occurred: #{e.message}"
  STDERR.puts e.backtrace.join("\n")
  exit 1
end

