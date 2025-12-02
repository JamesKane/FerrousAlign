#!/usr/bin/env ruby

# Parse existing benchmark logs and generate summary tables
# Usage: ruby parse_benchmark_logs.rb <bench_dir>

require 'time'

# Helper function to format numbers with comma separators
def format_number(num)
  num.to_s.reverse.gsub(/(\d{3})(?=\d)/, '\\1,').reverse
end

# Parse a log file and extract metrics
def parse_log(log_path, sam_path)
  metrics = {
    wall_time_s: nil,
    max_rss_kb: nil,
    user_time_s: nil,
    system_time_s: nil,
    simd_width: "N/A",
    simd_parallelism: "N/A",
    pairs_rescued: nil,
    exit_status: nil
  }

  if File.exist?(log_path)
    File.readlines(log_path).each do |line|
      # Parse /usr/bin/time -v output
      if line =~ /Maximum resident set size \(kbytes\):\s*(\d+)/
        metrics[:max_rss_kb] = $1.to_i
      elsif line =~ /User time \(seconds\):\s*([0-9\.]+)/
        metrics[:user_time_s] = $1.to_f
      elsif line =~ /System time \(seconds\):\s*([0-9\.]+)/
        metrics[:system_time_s] = $1.to_f
      elsif line =~ /Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:\.]+)/
        time_str = $1
        # Parse wall clock time (M:SS.ms or H:MM:SS)
        if time_str =~ /(\d+):(\d+)\.(\d+)/
          minutes = $1.to_i
          seconds = $2.to_f + ($3.to_f / 100.0)
          metrics[:wall_time_s] = minutes * 60 + seconds
        elsif time_str =~ /(\d+):(\d+):(\d+)\.(\d+)/
          hours = $1.to_i
          minutes = $2.to_i
          seconds = $3.to_f + ($4.to_f / 100.0)
          metrics[:wall_time_s] = hours * 3600 + minutes * 60 + seconds
        end
      elsif line =~ /Exit status:\s*(\d+)/ || line =~ /Command terminated by signal (\d+)/
        metrics[:exit_status] = $1.to_i
      end

      # Parse ferrous-align SIMD info
      if line =~ /Using compute backend:\s*([A-Z0-9\-]+)\s*\((\d+)-bit,\s*(\d+)-way parallelism\)/
        backend = $1
        bits = $2
        ways = $3
        metrics[:simd_width] = "#{bits}-bit (#{backend})"
        metrics[:simd_parallelism] = ways
      end

      # Parse pairs rescued
      if line =~ /([0-9,]+)\s+pairs rescued/i
        metrics[:pairs_rescued] = $1.delete(',').to_i
      end
    end
  end

  # Count records in SAM file
  metrics[:record_count] = 0
  if File.exist?(sam_path)
    metrics[:record_count] = `grep -v '^@' #{sam_path} | wc -l`.strip.to_i
  end

  # Run samtools flagstats
  if File.exist?(sam_path) && metrics[:record_count] > 0
    stdout_str = `samtools flagstats #{sam_path} 2>/dev/null`
    stdout_str.each_line do |line|
      if line =~ /\bmapped\b.*\(([^%]+)%/
        metrics[:mapped_percent] = $1.to_f
      elsif line =~ /\bproperly paired\b.*\(([^%]+)%/
        metrics[:properly_paired_percent] = $1.to_f
      end
    end
  end

  metrics
end

# Print detailed metrics table
def print_detailed_metrics(name, metrics, total_reads, total_pairs)
  puts ""
  puts "### #{name}"
  puts ""

  total_records = metrics[:record_count]
  supplementary = total_records - total_reads

  # CPU time and efficiency
  cpu_time = nil
  cpu_multiplier = nil
  if metrics[:user_time_s] && metrics[:system_time_s]
    cpu_time = metrics[:user_time_s] + metrics[:system_time_s]
    if metrics[:wall_time_s] && metrics[:wall_time_s] > 0
      cpu_multiplier = cpu_time / metrics[:wall_time_s]
    end
  end

  # Throughput
  throughput = nil
  if metrics[:wall_time_s] && metrics[:wall_time_s] > 0
    throughput = total_records / metrics[:wall_time_s]
  end

  # Wall time formatting
  wall_time_str = "N/A"
  if metrics[:wall_time_s]
    minutes = (metrics[:wall_time_s] / 60).to_i
    seconds = metrics[:wall_time_s] % 60
    wall_time_str = "#{minutes}:#{sprintf('%05.2f', seconds)} (#{sprintf('%.2f', metrics[:wall_time_s])}s)"
  end

  puts "| Metric | Value |"
  puts "|--------|-------|"
  puts "| **SIMD Width** | **#{metrics[:simd_width]}** |"
  puts "| **Parallelism** | **#{metrics[:simd_parallelism]}-way** |"
  puts "| Total reads | #{format_number(total_reads)} (#{format_number(total_pairs)} pairs) |"
  puts "| Total records | #{format_number(total_records)} |"
  puts "| Supplementary | #{format_number(supplementary)} |"

  if metrics[:mapped_percent]
    puts "| Mapped | #{sprintf('%.2f', metrics[:mapped_percent])}% |"
  else
    puts "| Mapped | N/A |"
  end

  if metrics[:properly_paired_percent]
    puts "| Properly paired | #{sprintf('%.2f', metrics[:properly_paired_percent])}% |"
  else
    puts "| Properly paired | N/A |"
  end

  if metrics[:pairs_rescued]
    puts "| Pairs rescued | #{format_number(metrics[:pairs_rescued])} |"
  end

  puts "| **Wall time** | **#{wall_time_str}** |"

  if cpu_time
    puts "| CPU time | #{sprintf('%.2f', cpu_time)}s |"
  else
    puts "| CPU time | N/A |"
  end

  if throughput
    puts "| **Throughput** | **#{format_number(throughput.to_i)} reads/sec** |"
  else
    puts "| **Throughput** | **N/A** |"
  end

  if cpu_multiplier
    cpu_eff_display = sprintf('%.0f%% (%.1fx parallel)', cpu_multiplier * 100, cpu_multiplier)
    puts "| CPU efficiency | #{cpu_eff_display} |"
  else
    puts "| CPU efficiency | N/A |"
  end

  if metrics[:max_rss_kb]
    puts "| Max memory | #{sprintf('%.1f', metrics[:max_rss_kb] / 1024.0 / 1024.0)} GB |"
  else
    puts "| Max memory | N/A |"
  end

  puts ""
end

# Main
bench_dir = ARGV[0] || '/home/jkane/benches'

TOTAL_READS = 8_000_000
TOTAL_PAIRS = 4_000_000

runs = [
  { name: "AVX2 on GRCh38", log: "#{bench_dir}/avx2_grch38.log", sam: "#{bench_dir}/avx2_grch38.sam" },
  { name: "AVX2 on CHM13v2.0", log: "#{bench_dir}/avx2_chm13.log", sam: "#{bench_dir}/avx2_chm13.sam" },
  { name: "BWA-MEM2 on GRCh38", log: "#{bench_dir}/bwamem2_grch38.log", sam: "#{bench_dir}/bwamem2_grch38.sam" },
]

results = []
runs.each do |run|
  metrics = parse_log(run[:log], run[:sam])
  results << { name: run[:name], metrics: metrics, log: run[:log], sam: run[:sam] }
end

# Print summary table
puts "### ⏱️ Alignment Performance"
puts "| Tool | Reference | SIMD | Wall Time | Max Memory (GB) | Throughput (reads/s) |"
puts "|:---|:---|:---:|:---:|:---:|:---:|"

results.each do |r|
  m = r[:metrics]
  next if m[:record_count] == 0  # Skip failed runs

  tool_name = r[:name].include?("BWA-MEM2") ? "BWA-MEM2" : "AVX2"
  reference_name = r[:name].include?("GRCh38") ? "GRCh38" : "CHM13v2.0"

  simd_label = m[:simd_width].split(' ').first

  wall_time_str = "N/A"
  if m[:wall_time_s]
    minutes = (m[:wall_time_s] / 60).to_i
    seconds = m[:wall_time_s] % 60
    wall_time_str = "#{minutes}:#{sprintf('%05.2f', seconds)}"
  end

  max_memory_gb = m[:max_rss_kb] ? sprintf('%.2f', m[:max_rss_kb] / 1024.0 / 1024.0) : "N/A"

  throughput = m[:record_count] > 0 && m[:wall_time_s] && m[:wall_time_s] > 0 ?
               (m[:record_count] / m[:wall_time_s]).to_i : 0

  puts "| **#{tool_name}** | #{reference_name} | #{simd_label} | #{wall_time_str} | #{max_memory_gb} | #{format_number(throughput)} |"
end

puts ""

# Print detailed metrics for each run
results.each do |r|
  next if r[:metrics][:record_count] == 0  # Skip failed runs
  print_detailed_metrics(r[:name], r[:metrics], TOTAL_READS, TOTAL_PAIRS)
end
