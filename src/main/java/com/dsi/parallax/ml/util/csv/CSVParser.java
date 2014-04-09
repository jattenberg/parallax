package com.dsi.parallax.ml.util.csv;

import java.util.List;
import java.util.Set;

/**
 * utility class for parsing various header files, generating CSV classes and
 * sources from the information in the header.
 * 
 * @author jattenberg
 * 
 */
public class CSVParser {

	public static CSV givenColumnsAndDataTypes(List<String> columns,
			List<Set<String>> dataTypes) {
		return CSV.columnNamesAndValues(columns, dataTypes);
	}

}
