/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.LinkedHashSet;
import java.util.Properties;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

public class FileUtil {


	public static void copyfile(String srcFile, String dstFile){
		try{
			File f1 = new File(srcFile);
			File f2 = new File(dstFile);
			InputStream in = new FileInputStream(f1);
			OutputStream out = new FileOutputStream(f2);

			byte[] buf = new byte[1024];
			int len;
			while ((len = in.read(buf)) > 0){
				out.write(buf, 0, len);
			}
			in.close();
			out.close();
		}
		catch(FileNotFoundException ex){
			ex.printStackTrace();
		}
		catch(IOException e){
			e.printStackTrace();
		}
	}

	/**
	 * Returns a BufferedReader to the file. Will unzip and gunzip depending on
	 * file extension.
	 * 
	 * @param filename
	 * @return
	 * @throws IOException
	 */
	public static BufferedReader getReader(String filename) throws IOException {
		return new BufferedReader(new InputStreamReader(getInputStream(filename)));
	}

	/**
	 * Returns an inputStream to a file.  Will unzip or gunzip the stream based on file extension.
	 * 
	 * @param filename
	 * @return
	 * @throws IOException
	 */
	public static InputStream getInputStream(String filename) throws IOException {
		InputStream in = new FileInputStream(filename);
		if (filename.endsWith(".gz")) {
			in = new GZIPInputStream(in);
		} else if (filename.endsWith(".zip")) {
			in = new ZipInputStream(in);
		}
		return in;
	}

	public static OutputStream getOutputStream(String filename) throws IOException {
		OutputStream out = new FileOutputStream(filename);
		if (filename.endsWith(".gz")) {
			out = new GZIPOutputStream(out);
		} else if (filename.endsWith(".zip")) {
			out = new ZipOutputStream(out);
		}
		return out;
	}


	public static OutputStream getOutputStream(File file) throws IOException {
		return getOutputStream(file.getAbsolutePath());
	}

	public static void writeObject(Object o, String filename) {
		ObjectOutput oo;
		try {
			oo = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream(filename)));
			oo.writeObject(o);
			oo.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			e.printStackTrace();
		} catch (IOException e) {

			e.printStackTrace();
		}
		// log.info("wrote object "+o.getClass().getName()+" to "+filename);
	}

	public static void writeObjectNoGzip(Object o, String filename) {
		ObjectOutput oo;
		try {
			oo = new ObjectOutputStream(new FileOutputStream(filename));
			oo.writeObject(o);
			oo.close();
		} catch (FileNotFoundException e) {

			e.printStackTrace();
		} catch (IOException e) {

			e.printStackTrace();
		}
		// log.info("wrote object "+o.getClass().getName()+" to "+filename);
	}

	public static Object readObject(String filename) {
		Object o = null;
		try {
			ObjectInput oo = new ObjectInputStream(new GZIPInputStream(new FileInputStream(filename)));

			o = oo.readObject();
			oo.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		return o;
	}

	public static String readFile(String filename) {
		StringBuffer buffer = null;

		if (filename != null) {
			try {
				buffer = new StringBuffer();
				BufferedReader br = new BufferedReader(new FileReader(filename));
				try {
					String line = null;
					while ((line = br.readLine()) != null) {
						buffer.append(line);
					}
				} finally {
					br.close();
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return ((buffer != null) ? buffer.toString() : null);
	}

	public static Properties getProperties(String propFile) {
		Properties props = new Properties();
		try {
			FileInputStream is = new FileInputStream(propFile);
			props.load(is);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return props;
	}

	/**
	 * Reads the reports config file and returns the Database fields as a 
	 * LinkedHashSet 
	 * 
	 * @param reportName - name of the report
	 * @param reportConfigFile - reportConfigFile used in hadoop process
	 * @return LinkedHashSet<String>  - a LinkedHashSet of the fields in the table
	 * @throws IOException
	 */
	public static LinkedHashSet<String> loadReportsConf(
			String reportConfigFile, String reportName) throws IOException{
		return loadReportsConf(new File(reportConfigFile), reportName);
	}

	public static LinkedHashSet<String> loadReportsConf(
			File reportConfigFile, String reportName) throws IOException{
		return loadReportsConf(new FileReader(reportConfigFile), reportName);
	}

	public static LinkedHashSet<String> loadReportsConf(
			FileReader reportConfigFile, String reportName) throws IOException  {
		LinkedHashSet<String> keys = new LinkedHashSet<String>();
		BufferedReader br = new BufferedReader(reportConfigFile);
		String lineStr = "";
		boolean readConfig = false;
		keys.add("HIT_DATE");
		keys.add("HIT_HOUR");
		while ((lineStr = br.readLine()) != null ){
			if(readConfig && !lineStr.startsWith("name")){
				if (!lineStr.startsWith("field")) continue;
				String[] key = lineStr.split("=");
				keys.add(key[2]);
			} else if (lineStr.startsWith("name")){
				readConfig = false;
			}
			if (lineStr.startsWith("name="+reportName)){
				readConfig = true;
			}
		}
		keys.add("CNT");
		return keys;

	}

	public static String getResource(InputStream is) {
		if (is != null) {
			StringBuilder sb = new StringBuilder();
			String line;

			try {
				BufferedReader reader = new BufferedReader(new InputStreamReader(is, "UTF-8"));
				while ((line = reader.readLine()) != null) {
					sb.append(line);
				}
			} catch (IOException e) {
				e.printStackTrace();
			} finally {
				try {
					is.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			return sb.toString();
		}
		System.err.println("not fount");
		return "";
	}
}
