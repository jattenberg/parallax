package com.dsi.parallax.ml.classifier.linear;

import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

import com.dsi.parallax.ml.vector.LinearVector;

/**
 * utility class for pretty-printing linear models
 * 
 * @author jattenberg
 * 
 */
public class LinearModelPrinter {

	public static <K, V> String prettyPrintMap(Map<K, V> map) {
		StringBuilder sb = new StringBuilder();
		Iterator<Entry<K, V>> iter = map.entrySet().iterator();
		while (iter.hasNext()) {
			Entry<K, V> entry = iter.next();
			sb.append(entry.getKey());
			sb.append('=').append('"');
			sb.append(entry.getValue());
			sb.append('"');
			if (iter.hasNext()) {
				sb.append(',').append(' ');
			}
		}
		return sb.toString();
	}

	public static String prettyPrintVector(LinearVector vec) {
		StringBuilder sb = new StringBuilder();
		Iterator<Integer> iter = vec.iterator();
		while (iter.hasNext()) {
			int x_i = iter.next();
			double y_i = vec.getValue(x_i);
			sb.append(x_i);
			sb.append('=').append(String.format("%.4f", y_i));
			if (iter.hasNext()) {
				sb.append(", ");
			}
		}

		return sb.toString();
	}
}
