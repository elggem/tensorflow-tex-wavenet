import fnmatch
import os
import threading

import numpy as np
import tensorflow as tf


def find_files(directory, pattern='*.csv'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def _read_lines(filename):
  with tf.gfile.GFile(filename, "r") as f:
    lines = f.read().decode("utf-8").split("\n")
    return lines

def load_csv(directory):
    '''Generator that yields text raw from the directory.'''
    # TODO: This can probably be way more efficient.
    files = find_files(directory)
    for filename in files:
        output = []
        lines = _read_lines(filename)
        for line in lines:
            if len(line)>0:
                line += ",1,1,1,1,1"
                line_val = np.array(line.split(","),dtype=np.float32)
                line_val = np.power(line_val, 0.5) * 1024 # scale up and apply gradient for quantization of network
                line_val *= 1024 # scale up and apply gradient for quantization of network
                output = np.append(output, line_val)
        
        yield output.reshape((-1, 1)) 
        #yield output


class CSVReader(object):
    '''Generic background text reader that preprocesses text files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 text_dir,
                 coord,
                 sample_size=None,
                 queue_size=256):
        self.text_dir = text_dir
        self.coord = coord
        self.sample_size = sample_size
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[(None, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def thread_main(self, sess):
        buffer_ = np.array([])
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_csv(self.text_dir)
            for data in iterator:
                #print("looking at " + str(data))
                if self.coord.should_stop():
                    self.stop_threads()
                    stop = True
                    break
                if self.sample_size:
                    # Cut samples into fixed size pieces
                    buffer_ = np.append(buffer_, data)
                    while len(buffer_) > self.sample_size:
                        piece = np.reshape(buffer_[:self.sample_size], [-1, 1])
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece})
                        buffer_ = buffer_[self.sample_size:]
                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: data})

    def stop_threads():
        for t in self.threads:
            t.stop()

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
