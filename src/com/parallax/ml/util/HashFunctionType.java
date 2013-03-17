/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.util;

import java.security.MessageDigest;


public enum HashFunctionType
{
	SHA {
		@Override
		public int hash(String input, int bins)
		{
			byte[] bytes = sha.digest(input.getBytes());
			int out = 17;
			for(byte b : bytes)
				out = 31*out+b;
			return Math.abs(out)%bins;
		}
	},
	MD5
	{
		
		@Override
		public int hash(String input, int bins)
		{
			byte[] bytes = md5.digest(input.getBytes());
			int out = 17;
			for(byte b : bytes)
				out = 31*out+b;
			return Math.abs(out)%bins;
		}
	},
	JENKINS
	{
		@Override
		public int hash(String token, int bins)
		{
			// alternate hash function.
			// slower then String.hashCode(), but possibly is a more uniform
			// hash
			// function
			byte[] input = token.getBytes();
			return jenkinsHashOnBytes(bins, input);
		}
	},
	JAVA
	{
		@Override
		public int hash(String input, int bins)
		{
			return Math.abs((input).hashCode()) % bins;
		}
	};

	static MessageDigest md5, sha;
	static
	{
		try
		{
			md5 = MessageDigest.getInstance("MD5");
			sha = MessageDigest.getInstance("SHA-256");
		} catch (Exception e)
		{
			// TODO: handle exception
		}
		
	}
	public abstract int hash(String input, int bins);
	private static int jenkinsHashOnBytes(int bins, byte[] input)
	{
		long hash = 0;
		String key = new String(input);
		for (int i = 0; i < key.length(); i++)
		{
			hash += key.charAt(i);
			hash += (hash << 10);
			hash ^= (hash >> 6);
		}

		hash += (hash << 3);
		hash ^= (hash >> 11);
		hash += (hash << 15);
		return (int) Math.abs(hash % bins);
	}
}
