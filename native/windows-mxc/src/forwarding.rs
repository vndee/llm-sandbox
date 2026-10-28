//! Preserve output bytes and promptly publish chunks that contain no newline.

use std::io::{self, Read, Write};

pub fn copy_and_flush(reader: &mut dyn Read, writer: &mut dyn Write) -> io::Result<()> {
    let mut buffer = [0; 8192];
    loop {
        let length = match reader.read(&mut buffer) {
            Ok(0) => return Ok(()),
            Ok(length) => length,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        // Stdout's line buffering must not delay callback data until newline or
        // process exit. Do not hold a global stdio lock across the blocking read;
        // SDK teardown may need stderr concurrently on an error path.
        writer.write_all(&buffer[..length])?;
        writer.flush()?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    #[test]
    fn publishes_binary_chunks_before_reading_again_without_waiting_for_eof() {
        struct Input {
            reads: usize,
            flushes: Rc<Cell<usize>>,
        }
        impl Read for Input {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                assert_eq!(self.flushes.get(), self.reads);
                if self.reads == 2 {
                    return Ok(0);
                }
                buffer[..3].copy_from_slice(&[0xff, 0, self.reads as u8]);
                self.reads += 1;
                Ok(3)
            }
        }
        struct Output {
            bytes: Vec<u8>,
            flushes: Rc<Cell<usize>>,
        }
        impl Write for Output {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.bytes.extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                self.flushes.set(self.flushes.get() + 1);
                Ok(())
            }
        }
        let flushes = Rc::new(Cell::new(0));
        let mut input = Input {
            reads: 0,
            flushes: flushes.clone(),
        };
        let mut output = Output {
            bytes: Vec::new(),
            flushes,
        };
        copy_and_flush(&mut input, &mut output).unwrap();
        assert_eq!(output.bytes, [0xff, 0, 0, 0xff, 0, 1]);
    }

    #[test]
    fn forwarding_errors_are_returned_for_process_tree_cancellation() {
        struct Closed;
        impl Write for Closed {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let result = copy_and_flush(&mut &b"progress"[..], &mut Closed);
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::BrokenPipe);
    }
}
