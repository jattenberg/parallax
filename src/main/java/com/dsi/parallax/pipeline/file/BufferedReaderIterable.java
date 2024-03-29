/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.file;

import org.apache.log4j.Logger;

import java.io.*;
import java.util.Iterator;

/**
 * BufferedReaderIterable reads File object into the BufferedReader object, 
 * it will build Iterator object to encapsulate BufferedReader object; 
 * then user can use iterator's hasNext() to check if read data
 * from BufferedReader, and use iterator's next() to read and get data from 
 * BufferedReader
 *
 * @author Josh Attenberg
 */
public class BufferedReaderIterable implements Iterable<String> {
    private Iterator<String> iterator;
    private static final Logger LOGGER = Logger
            .getLogger(BufferedReaderIterable.class);

    /**
     * Class constructor specifying bufferedReader object to create
     * @param br bufferedReader object
     */
    public BufferedReaderIterable(BufferedReader br) {
        iterator = new BufferedReaderIterator(br);
    }

    /**
     * Class constructor specifying file object to create
     * @param f file
     * @throws FileNotFoundException  file not found exception
     */
    public BufferedReaderIterable(File f) throws FileNotFoundException {
        this(new BufferedReader(new FileReader(f)));
    }

    @Override
	public Iterator<String> iterator() {
        return iterator;
    }

    private class BufferedReaderIterator implements Iterator<String> {
        private BufferedReader reader;
        private String line;

        public BufferedReaderIterator(BufferedReader reader) {
            this.reader = reader;
            advance();
        }

        @Override
		public boolean hasNext() {
            return line != null;
        }

        @Override
		public String next() {
            String retval = line;
            advance();
            return retval;
        }

        @Override
		public void remove() {
            throw new UnsupportedOperationException(
                    "Remove not supported on BufferedReader iteration.");
        }

        private void advance() {
            try {
                line = reader.readLine();
            } catch (IOException e) {
                LOGGER.error(e.getLocalizedMessage());
            }
            if (line == null && reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    LOGGER.error(e.getLocalizedMessage());
                }
                reader = null;
            }
        }
    }
}
