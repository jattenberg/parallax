/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.file;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileChannel.MapMode;
import java.util.Iterator;
import java.util.LinkedList;

import org.apache.commons.lang.StringUtils;

import com.google.common.io.Files;

/**
 * MemoryMappedFileReaderIterable specializes in reading <b>large</b> files, typically in multiple 
 * GBs, by first mapping the file into Direct Byte Buffer, or memory mapping the file, and then
 * reading values from the local memory instead of doing disc seeks. Reading file by first 
 * memory mapping the file is the fastest file read option available in java. 
 * <p>
 * To use this class, you need to add <b>-XX:MaxDirectMemorySize=XG</b>, where X is close to 
 * the size of file to be read in GB, to java vm options. 
 * <p>
 * 
 * @author Rahul Ratnakar
 */
public class MemoryMappedFileReaderIterable implements Iterable<String> {
    private Iterator<String> iterator;
    static final int MAX_BUFFER_SIZE = Integer.MAX_VALUE; // This is 2GB. 
    static final String NEW_LINE = System.getProperty("line.separator");
    ByteBuffer[] buffArray = null;

    /**
     * Class constructor specifying file to read.
     * @param f file, preferably a very large file.
     * @throws IOException 
     */
	public MemoryMappedFileReaderIterable(File f) throws IOException {
		FileChannel channel = new RandomAccessFile(f, "r").getChannel();
		long fileLength = f.length();
		int buffArraySize = 1;
		if (fileLength > MAX_BUFFER_SIZE) {
			buffArraySize = (int) (fileLength / MAX_BUFFER_SIZE) + 1;
		}
		buffArray = new ByteBuffer[buffArraySize];
		long start = 0;
		for(int i = 0; i < buffArraySize ; i++ ) {
			buffArray[i] = channel.map(MapMode.READ_ONLY, start, Math.min((fileLength - start) , MAX_BUFFER_SIZE ));
			start = start + MAX_BUFFER_SIZE;
		}
		iterator = new MemoryMappedFileReaderIterator();
	}

    @Override
	public Iterator<String> iterator() {
        return iterator;
    }
    
	public static void main(String[] args) throws IOException {
		String file = args[0];
		if(StringUtils.isNotBlank(file)) {
			String mmFileOut = file + ".mm";
			String buffFileOut = file + ".buff";
			BufferedWriter mmOut = new BufferedWriter(new FileWriter(new File(mmFileOut)));
			try {
				long start = System.nanoTime();
				Iterable<String> itr = new MemoryMappedFileReaderIterable(new File(file));
				while (itr.iterator().hasNext()) {
					mmOut.write(itr.iterator().next());
				}
				long mmTime = System.nanoTime() - start;
				mmOut.close();

				BufferedWriter buffOut = new BufferedWriter(new FileWriter(new File(buffFileOut)));

				start = System.nanoTime();
				itr = new BufferedReaderIterable(new File(file));
				while (itr.iterator().hasNext()) {
					buffOut.write(itr.iterator().next());
				}
				long buffTime = System.nanoTime() - start;
				buffOut.close();
				System.out.println("File size " + new File(file).length());
				long mmchecksum = Files.getChecksum(new File(mmFileOut), new java.util.zip.CRC32());
				long buffchecksum = Files.getChecksum(new File(buffFileOut), new java.util.zip.CRC32());
				System.out.println("MemoryMapped :- " + "Time = " + mmTime + " , chksum = " +  mmchecksum);
				System.out.println("Buffered :- " + "Time = " + buffTime + " , chksum = " +  buffchecksum);
				if(mmchecksum == buffchecksum) {
					System.out.println(mmTime <= buffTime ? "Memory mapped is better by " + (buffTime - mmTime) + " ns." : "Bufffered is better by " + (mmTime - buffTime) + " ns.");
				} else {
					System.out.println("ERROR!! Checksum does not match");
				}
				new File(mmFileOut).delete();
				new File(buffFileOut).delete();
			} catch (Exception e) {
				e.printStackTrace();
			}
			System.out.println("Done");
		}
		
	}

    private class MemoryMappedFileReaderIterator implements Iterator<String> {
        private String line ;
        final int SIZE = 8*1024; // make it configurable? 
        byte[] tempArray; 
        LinkedList<String>  readStrings = new LinkedList<String>();
        String carryOverString = "";
        
        int readStringCurPos = 0;
        int readStringLength = 0;
        
        int currentBufferIndex = 0 ;
        byte[] tempChars = new byte[NEW_LINE.length()];

        public MemoryMappedFileReaderIterator() {
        	if(buffArray != null)
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
			if((currentBufferIndex < buffArray.length) &&  (readStringCurPos >= readStringLength)) {
				int remaining = buffArray[currentBufferIndex].remaining(); 
				if(remaining == 0 && ((currentBufferIndex + 1) < buffArray.length)) {
					buffArray[currentBufferIndex] = null;
					currentBufferIndex++;
				}
				int toCopy = Math.min(buffArray[currentBufferIndex].remaining(),  SIZE) ;
				if(toCopy > 0) {
					tempArray = new byte[toCopy]; 
					buffArray[currentBufferIndex].get(tempArray,0,toCopy);
					split(new String(tempArray),NEW_LINE,readStrings);
					System.arraycopy(tempArray, 0 , tempChars, 0, NEW_LINE.length());
					if(new String(tempChars).equalsIgnoreCase(NEW_LINE)) {
						readStrings.add(0, carryOverString);
					} else {
						readStrings.set(0, carryOverString + readStrings.get(0)) ;
					}
					System.arraycopy(tempArray, tempArray.length - NEW_LINE.length() , tempChars, 0, NEW_LINE.length());
					if(!new String(tempChars).equalsIgnoreCase(NEW_LINE)) {
						carryOverString = readStrings.get(readStrings.size()-1);
						readStringLength = readStrings.size()-1;
					} else {
						carryOverString = "";
						readStringLength = readStrings.size();
					}
					readStringCurPos = 0;
				}
			}
			if(readStringCurPos < readStringLength) {
				line = readStrings.get(readStringCurPos++);
			} else {
				if(StringUtils.isNotEmpty(carryOverString)) 
				{
					line = carryOverString;
					carryOverString = null;
				} else {
					line = null;
				}
			}
		}
		
		private void split(String str, String separatorChars,LinkedList<String> list) {
			list.clear();
			if (StringUtils.isNotBlank(str) && StringUtils.isNotEmpty(separatorChars)) {
				int len = str.length();
				int i = 0, start = 0;
				boolean match = false;
				if (separatorChars.length() == 1) {

					char sep = separatorChars.charAt(0);
					while (i < len) {
						if (str.charAt(i) == sep) {
							if (match) {
								list.add(str.substring(start, i));
								match = false;
							}
							start = ++i;
							continue;
						} 
						match = true;
						i++;
					}
				} else {
					while (i < len) {
						if (separatorChars.indexOf(str.charAt(i)) >= 0) {
							if (match) {
								list.add(str.substring(start, i));
								match = false;
							}
							start = ++i;
							continue;
						} 
						match = true;
						i++;
					}
				}
				if (match) {
					list.add(str.substring(start, i));
				}
			}
		}
    }
}
