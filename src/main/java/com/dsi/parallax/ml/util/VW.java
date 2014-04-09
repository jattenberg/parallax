/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util;

import java.util.HashMap;
import java.util.Map;

/**
 * file format i often use for representing namespaced data
 * TODO: numeric features
 * TODO: pipe methods
 * 
 * @author jattenberg
 *
 */
public class VW
{
	private Map<String, String> namespaceData;
	private double label = -1;
	private String category = null;
	private String id = null;
	
	public VW()
	{
		namespaceData = new HashMap<String, String>();
	};
	
	public void addData(String name, String vwd)
	{
		namespaceData.put(name, vwd);
	}
	public static VW getVW(String id, String category)
	{
		VW vw = new VW();
		vw.category = category;
		vw.id = id;
		return vw;
	}
	public static VW fromVWLine(String line)
	{
		return fromLine(line,false);
	}
	public static VW fromTVWline(String line)
	{
		return fromLine(line, true);
	}
	
	/**
	 * TODO: numeric data 
	 * @param line
	 * @param category
	 * @return
	 */
	public static VW fromLine(String line, boolean category)
	{
		VW vw = new VW();
		int start=0, end;
		end = line.indexOf(" ");
		int tmp = line.indexOf("|");
		if(tmp==-1)
			return vw;
		end = line.substring(0, tmp).lastIndexOf(" ");
		if(end==-1)
			return vw;
		if(category)
			vw.category = line.substring(start, end);
		else
			vw.label = vw.label = Double.parseDouble(line.substring(start, end));
		start = end;
		end = line.indexOf("|");

		if(end==-1)
			return vw;
		vw.id = line.substring(start+1, end);
		
		start = end;
		String[] parts = line.substring(start+1).split("\\|");
		int i=0;
		for(String part : parts)
		{
			start = 0;
			end = part.indexOf(" ");
			if(end==-1)
				continue;
			String namespace = (end!=0)?part.substring(start,end):i+"";
			vw.addData(namespace, part.substring(end+1));
			i++;
		}

		return vw;
	}
	public String getCategory()
	{
		return category==null ? label+"" : category;
	}
	public String getID()
	{
		return id;
	}
	public Map<String,String> getNamespaceData()
	{
		return namespaceData;
	}
	
	@Override
	public String toString()
	{
		StringBuilder buff = new StringBuilder(id+", label: "+ label + ", category: " + category);
		for(String ns : namespaceData.keySet())
			buff.append(", " + ns+"->"+namespaceData.get(ns));
		return buff.toString();
	}

	public void setId(String id) {
		this.id = id;
	}

	public void setCategory(String category) {
		this.category = category;
	}
	public void addNamespaceData(String namespace, String data)
	{
		namespaceData.put(namespace, data);
	}
	public void addNamespaceData(Map<String,String> data)
	{
		namespaceData.putAll(data);
	}

	
}
