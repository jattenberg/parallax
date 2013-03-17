package com.parallax.ml.util.csv;

import static com.google.common.base.Preconditions.checkArgument;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;

import org.apache.commons.lang.StringUtils;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.parallax.ml.vector.LinearVector;
import com.parallax.ml.vector.LinearVectorFactory;
import com.parallax.pipeline.Context;

/**
 * an attempt at the simplest possible flexible CSV implementation. used for
 * buidling an understanding for the values present in each column of a csv.
 * 
 * @author jattenberg
 */
public class CSV implements Serializable {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -4424729102725186132L;

	/**
	 * The Constant NUMERIC_VALUE. a flag indicating that a column is numeric
	 */
	private static final Set<String> NUMERIC_VALUE = Collections
			.singleton("___NUMERIC___");

	/** The Constant COL_NAME. */
	private static final String COL_NAME = "COLUMN____";

	/** The Constant VAL_NAME. */
	private static final String VAL_NAME = "____VALUE____";

	/** The num columns. how many columns are in the CSV considered */
	private int numColumns;

	/** Mapping the names of columns to their respective index, and vice-versa */
	private final BiMap<String, Integer> nameToColumn;

	/** Storing the types of values to be expected in each column */
	private final List<Set<String>> columnValues;

	/** How do we map the values present in a column to an output dimension */
	private final Map<String, Integer> valueToOutputDimension;

	/** The size of the output */
	private int outputDimension;

	/** set of columns to ignore */
	private final Set<String> ignoreColumns;

	/**
	 * The column storing the label value in the data. -1 flag indicates unused.
	 */
	private int labelColumn = -1;

	/**
	 * The id column storing the identifier of a particular column. -1 flag
	 * indicates unused
	 */
	private int idColumn = -1;

	/** has the value to output dimension data structure been initialized? */
	private boolean initialized;

	/** What value to use in the event of missing data? */
	private double missingValue = 0;

	/**
	 * 
	 */
	private CSV() {
		nameToColumn = HashBiMap.create(numColumns);
		columnValues = Lists.newArrayListWithCapacity(numColumns);
		valueToOutputDimension = Maps.newHashMap();
		ignoreColumns = Sets.newHashSet();
		outputDimension = 0;
	}

	/**
	 * default constrctor. sets up all data structures.
	 * 
	 * @param numColumns
	 *            the num columns
	 */
	private CSV(int numColumns) {
		checkArgument(numColumns > 0,
				"number of columns must be positive, given: %s", numColumns);

		this.numColumns = numColumns;
		nameToColumn = HashBiMap.create(numColumns);
		columnValues = Lists.newArrayListWithCapacity(numColumns);
		valueToOutputDimension = Maps.newHashMap();
		ignoreColumns = Sets.newHashSet();
		outputDimension = 0;
	}

	/**
	 * creates a "blank csv" to be populated with "adder" methods.
	 * 
	 * @return the csv
	 */
	public static CSV blankCSV() {
		return new CSV();
	}

	/**
	 * Builds a csv where all columns are numeric.
	 * 
	 * @param numColumns
	 *            the num columns
	 * @return the csv
	 */
	public static CSV numericColumnCSV(int numColumns) {
		CSV out = new CSV(numColumns);
		out.createIndexNames();
		out.populateNumeric();
		return out;
	}

	/**
	 * Builds a csv where all columns are numeric, with supplied column ids.
	 * 
	 * @param columnNames
	 *            the column names
	 * @return the csv
	 */
	public static CSV numericColumnCSV(List<String> columnNames) {
		checkArgument(columnNames != null && columnNames.size() > 0,
				"no column names given!");
		CSV out = new CSV(columnNames.size());
		out.populateNames(columnNames);
		out.populateNumeric();
		return out;
	}

	/**
	 * Sets up a CSV by the values contained in the columns
	 * 
	 * @param columnValues
	 *            the column values
	 * @return the csv
	 */
	public static CSV columnsByValue(List<Set<String>> columnValues) {
		checkArgument(columnValues != null && columnValues.size() > 0,
				"no column values given!");
		CSV out = new CSV(columnValues.size());
		out.createIndexNames();
		out.populateValues(columnValues);
		return out;
	}

	/**
	 * Sets up a CSV with the specified column names and column values.
	 * 
	 * @param columnNames
	 *            the column names
	 * @param columnValues
	 *            the column values
	 * @return the csv
	 */
	public static CSV columnNamesAndValues(List<String> columnNames,
			List<Set<String>> columnValues) {
		checkArgument(columnValues != null && columnValues.size() > 0,
				"no column values given!");
		checkArgument(columnNames != null && columnNames.size() > 0,
				"no column names given!");
		checkArgument(
				columnNames.size() == columnValues.size(),
				"number of names and types must match! names has size: %s, types has size: %s",
				columnNames.size(), columnValues.size());
		CSV out = new CSV(columnValues.size());
		out.populateNames(columnNames);
		out.populateValues(columnValues);
		return out;
	}

	/**
	 * Initialize the data structures and prepares the CSV for use
	 * 
	 * @return the csv
	 */
	public CSV initialize() {
		outputDimension = 0;
		for (int i = 0; i < columnValues.size(); i++) {
			if ((idColumn >= 0 && i == idColumn)
					|| (labelColumn >= 0 && i == labelColumn)
					|| ignoreColumns.contains(nameToColumn.inverse().get(i))) {
				continue;
			} else if (columnValues.get(i) == NUMERIC_VALUE) {
				valueToOutputDimension.put(nameToColumn.inverse().get(i),
						outputDimension++);
			} else {
				for (String value : columnValues.get(i)) {
					valueToOutputDimension.put(nameToColumn.inverse().get(i)
							+ VAL_NAME + value, outputDimension++);
				}
			}
		}
		initialized = true;
		return this;
	}

	/**
	 * Sets the missing value, what value should be used if the data in a
	 * numeric column is missing?
	 * 
	 * @param value
	 *            the value to be used
	 * @return the csv
	 */
	public CSV setMissingValue(double value) {
		missingValue = value;
		return this;
	}

	/**
	 * Sets the label column- which column stores the labels used for training
	 * or evaluating predictive models.
	 * 
	 * @param columnId
	 *            the column id
	 * @return the csv
	 */
	public CSV setLabelColumn(String columnId) {
		if (!nameToColumn.containsKey(columnId)) {
			throw new NoSuchElementException(columnId
					+ " isn't a valid column name.");
		}
		labelColumn = nameToColumn.get(columnId);
		if (idColumn >= 0 && labelColumn == idColumn) {
			throw new IllegalArgumentException(
					"label column must be different from Id column");
		}
		initialized = false;
		return this;
	}

	/**
	 * Sets the label column- which column stores the labels used for training
	 * or evaluating predictive models.
	 * 
	 * @param index
	 *            the index
	 * @return the csv
	 */
	public CSV setLabelColumn(int index) {
		if (index >= numColumns) {
			throw new ArrayIndexOutOfBoundsException("attempting to query "
					+ index + " when max size is: " + numColumns);
		}
		labelColumn = index;
		if (idColumn >= 0 && labelColumn == idColumn) {
			throw new IllegalArgumentException(
					"label column must be different from Id column");
		}
		if (ignoreColumns.contains(nameToColumn.inverse().get(index))) {
			ignoreColumns.remove(nameToColumn.inverse().get(index));
		}
		initialized = false;
		return this;
	}

	/**
	 * Sets the id column- which column can be used to identify an output
	 * instance
	 * 
	 * @param columnId
	 *            the column id
	 * @return the csv
	 */
	public CSV setIdColumn(String columnId) {
		if (!nameToColumn.containsKey(columnId)) {
			throw new NoSuchElementException(columnId
					+ " isn't a valid column name.");
		}
		idColumn = nameToColumn.get(columnId);
		if (labelColumn >= 0 && labelColumn == idColumn) {
			throw new IllegalArgumentException(
					"label column must be different from Id column");
		}
		if (ignoreColumns.contains(columnId)) {
			ignoreColumns.remove(columnId);
		}
		initialized = false;
		return this;
	}

	/**
	 * Sets the id column- which column can store the output instance
	 * 
	 * @param index
	 *            the index
	 * @return the csv
	 */
	public CSV setIdColumn(int index) {
		if (index >= numColumns) {
			throw new ArrayIndexOutOfBoundsException("attempting to query "
					+ index + " when max size is: " + numColumns);
		}
		idColumn = index;
		if (labelColumn >= 0 && labelColumn == idColumn) {
			throw new IllegalArgumentException(
					"label column must be different from Id column");
		}
		if (ignoreColumns.contains(nameToColumn.inverse().get(index))) {
			ignoreColumns.remove(nameToColumn.inverse().get(index));
		}
		initialized = false;
		return this;
	}

	/**
	 * Sets the id column- which column can be used to identify an output
	 * instance
	 * 
	 * @param columnId
	 *            the column id
	 * @return the csv
	 */
	public CSV ignoreColumn(String columnId) {
		if (!nameToColumn.containsKey(columnId)) {
			throw new NoSuchElementException(columnId
					+ " isn't a valid column name.");
		}
		int column = nameToColumn.get(columnId);

		if (labelColumn >= 0 && labelColumn == column) {
			throw new IllegalArgumentException("can't ignore labelColumn!");
		}

		if (idColumn >= 0 && idColumn == column) {
			throw new IllegalArgumentException("can't ignore idColumn!");
		}

		ignoreColumns.add(columnId);

		initialized = false;
		return this;
	}

	/**
	 * Sets the id column- which column can store the output instance
	 * 
	 * @param index
	 *            the index
	 * @return the csv
	 */
	public CSV ignoreColumn(int index) {
		if (index >= numColumns) {
			throw new ArrayIndexOutOfBoundsException("attempting to query "
					+ index + " when max size is: " + numColumns);
		}

		if (labelColumn >= 0 && labelColumn == index) {
			throw new IllegalArgumentException("can't ignore labelColumn!");
		}

		if (idColumn >= 0 && idColumn == index) {
			throw new IllegalArgumentException("can't ignore idColumn!");
		}

		ignoreColumns.add(nameToColumn.inverse().get(index));
		initialized = false;
		return this;
	}

	/**
	 * Populate values- stores the values that are expected in all columns
	 * 
	 * @param colVals
	 *            the col vals
	 * @return the csv
	 */
	private CSV populateValues(List<Set<String>> colVals) {

		for (int i = 0; i < colVals.size(); i++) {
			columnValues.add(colVals.get(i));
			if (colVals.get(i) == NUMERIC_VALUE) {
				valueToOutputDimension.put(nameToColumn.inverse().get(i),
						outputDimension++);
			} else {
				for (String value : colVals.get(i)) {
					valueToOutputDimension.put(nameToColumn.inverse().get(i)
							+ VAL_NAME + value, outputDimension++);
				}
			}
		}
		ignoreColumns.clear();
		initialized = true;
		return this;
	}

	/**
	 * Sets the values for categorical column.
	 * 
	 * @param columnId
	 *            the column id
	 * @param values
	 *            the values
	 * @return the csv
	 */
	public CSV setValuesForCategoricalColumn(String columnId, Set<String> values) {
		if (!nameToColumn.containsKey(columnId)) {
			throw new NoSuchElementException(columnId
					+ " isn't a valid column name.");
		}
		columnValues.set(nameToColumn.get(columnId), values);
		if (ignoreColumns.contains(columnId)) {
			ignoreColumns.remove(columnId);
		}
		initialized = false;
		return this;
	}

	/**
	 * Sets the values for categorical column.
	 * 
	 * @param index
	 *            the index
	 * @param values
	 *            the values
	 * @return the csv
	 */
	public CSV setValuesForCategoricalColumn(int index, Set<String> values) {
		if (index >= numColumns) {
			throw new ArrayIndexOutOfBoundsException("attempting to query "
					+ index + " when max size is: " + numColumns);
		}
		columnValues.set(index, values);
		if (ignoreColumns.contains(nameToColumn.inverse().get(index))) {
			ignoreColumns.remove(nameToColumn.inverse().get(index));
		}
		initialized = false;
		return this;
	}

	/**
	 * Sets the column numeric.
	 * 
	 * @param columnId
	 *            the column id
	 * @return the csv
	 */
	public CSV setColumnNumeric(String columnId) {
		if (!nameToColumn.containsKey(columnId)) {
			throw new NoSuchElementException(columnId
					+ " isn't a valid column name.");
		}
		columnValues.set(nameToColumn.get(columnId), NUMERIC_VALUE);
		if (ignoreColumns.contains(columnId)) {
			ignoreColumns.remove(columnId);
		}
		initialized = false;
		return this;
	}

	/**
	 * Sets the column numeric.
	 * 
	 * @param index
	 *            the index
	 * @return the csv
	 */
	public CSV setColumnNumeric(int index) {
		if (index >= numColumns) {
			throw new ArrayIndexOutOfBoundsException("attempting to query "
					+ index + " when max size is: " + numColumns);
		}
		columnValues.set(index, NUMERIC_VALUE);
		if (ignoreColumns.contains(nameToColumn.inverse().get(index))) {
			ignoreColumns.remove(nameToColumn.inverse().get(index));
		}
		initialized = false;
		return this;
	}

	/**
	 * Populate names.
	 * 
	 * @param columnNames
	 *            the column names
	 * @return the csv
	 */
	private CSV populateNames(List<String> columnNames) {
		for (int i = 0; i < columnNames.size(); i++) {
			nameToColumn.put(columnNames.get(i), i);
		}
		ignoreColumns.clear();
		initialized = false;
		return this;
	}

	/**
	 * Populate numeric.
	 * 
	 * @return the csv
	 */
	private CSV populateNumeric() {
		for (int i = 0; i < numColumns; i++) {
			columnValues.add(NUMERIC_VALUE);
			valueToOutputDimension.put(nameToColumn.inverse().get(i),
					outputDimension++);
		}
		initialized = true;
		return this;
	}

	/**
	 * Creates the index names.
	 * 
	 * @return the csv
	 */
	private CSV createIndexNames() {
		for (int i = 0; i < numColumns; i++) {
			String name = COL_NAME + i;
			nameToColumn.put(name, i);
		}
		initialized = false;
		return this;
	}

	/**
	 * Gets the name from index.
	 * 
	 * @param index
	 *            the index
	 * @return the name from index
	 */
	public String getNameFromIndex(int index) {
		if (!nameToColumn.inverse().containsKey(index)) {
			throw new ArrayIndexOutOfBoundsException("attempting to query "
					+ index + " when max size is: " + numColumns);
		}
		return nameToColumn.inverse().get(index);
	}

	/**
	 * Gets the index for column name.
	 * 
	 * @param name
	 *            the name
	 * @return the index for column name
	 */
	public int getIndexForColumnName(String name) {
		if (!nameToColumn.containsKey(name)) {
			throw new NoSuchElementException(name
					+ " isn't a valid column name.");
		}
		return nameToColumn.get(name);
	}

	/**
	 * Gets the num columns.
	 * 
	 * @return the num columns
	 */
	public int getNumColumns() {
		return numColumns;
	}

	/**
	 * Numeric values.
	 * 
	 * @return the sets the
	 */
	public static Set<String> numericValues() {
		return NUMERIC_VALUE;
	}

	// TODO: different ways of handling unknown feature value.
	/**
	 * Parses the row.
	 * 
	 * @param row
	 *            the row
	 * @return the context
	 */
	public Context<LinearVector> parseRow(final List<String> row) {
		if (!initialized) {
			throw new IllegalStateException(
					"CSV must be initialized before parsing data");
		}

		checkArgument(
				row.size() == numColumns,
				"size of input row must match the number of columns in CSV, given: %s, expecting: %s",
				row.size(), numColumns);

		LinearVector vect = LinearVectorFactory.getDenseVector(outputDimension);
		String id = null;
		String label = null;

		for (int i = 0; i < row.size(); i++) {
			if (idColumn >= 0 && i == idColumn) {
				id = row.get(i);
			} else if (labelColumn >= 0 && i == labelColumn) {
				label = row.get(i);
			} else if (ignoreColumns.contains(nameToColumn.inverse().get(i))) {
				continue;
			} else {

				Set<String> values = columnValues.get(i);
				if (values == NUMERIC_VALUE) {

					int dimension = valueToOutputDimension.get(nameToColumn
							.inverse().get(i));
					if (StringUtils.isBlank(row.get(i))) {
						vect.resetValue(dimension, missingValue);
					} else {
						vect.resetValue(dimension,
								Double.parseDouble(row.get(i)));
					}
				} else {
					String value = row.get(i);
					String key = nameToColumn.inverse().get(i) + VAL_NAME
							+ value;

					// ignore unknown feature values
					if (valueToOutputDimension.containsKey(key)) {
						vect.resetValue(valueToOutputDimension.get(key), 1d);
					}
				}
			}
		}

		return Context.createContext(id, label, vect);
	}
}
