use std::io::{Result as IoResult, Write};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::{self, JoinHandle};

// Messages sent to the background writer thread
enum OutMsg {
    Bytes(Vec<u8>),
    Flush,
    End,
}

/// AsyncChannelWriter: implements Write by enqueueing bytes to a background writer thread
///
/// Usage:
/// - Wrap any `Box<dyn Write + Send>` with `AsyncChannelWriter::new(inner)`
/// - The returned object also implements `Write`, so existing code continues to work
/// - On drop, it ensures pending data is flushed and the thread is joined
pub struct AsyncChannelWriter {
    tx: SyncSender<OutMsg>,
    handle: Option<JoinHandle<()>>,
}

impl AsyncChannelWriter {
    /// Create a new asynchronous writer, owning the provided inner writer.
    pub fn new(mut inner: Box<dyn Write + Send>) -> Self {
        // Bounded channel provides backpressure; tune capacity as needed
        let (tx, rx): (SyncSender<OutMsg>, Receiver<OutMsg>) = sync_channel(8192);

        let handle = thread::spawn(move || {
            // Writer loop: forward messages to the inner writer
            while let Ok(msg) = rx.recv() {
                match msg {
                    OutMsg::Bytes(buf) => {
                        if inner.write_all(&buf).is_err() {
                            // On error, attempt to break the loop; dropping rx will stop
                            break;
                        }
                    }
                    OutMsg::Flush => {
                        let _ = inner.flush();
                    }
                    OutMsg::End => {
                        let _ = inner.flush();
                        break;
                    }
                }
            }
        });

        Self {
            tx,
            handle: Some(handle),
        }
    }
}

impl Write for AsyncChannelWriter {
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        // Clone into Vec for transfer to the writer thread
        // Note: This copies; Phase B can reduce overhead by formatting in workers
        let _ = self.tx.send(OutMsg::Bytes(buf.to_vec()));
        Ok(buf.len())
    }

    fn flush(&mut self) -> IoResult<()> {
        let _ = self.tx.send(OutMsg::Flush);
        Ok(())
    }
}

impl Drop for AsyncChannelWriter {
    fn drop(&mut self) {
        // Signal end and join the writer thread
        let _ = self.tx.send(OutMsg::End);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
