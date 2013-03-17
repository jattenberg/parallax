/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.csv;

import java.lang.reflect.Type;
import java.util.Map;

import com.google.gson.reflect.TypeToken;
import com.parallax.ml.vector.LinearVector;
import com.parallax.ml.vector.LinearVectorFactory;
import com.parallax.pipeline.AbstractPipe;
import com.parallax.pipeline.Context;

/**
 * NumericCSVtoLabeledVectorPipe converts the String contexts into LinearVector
 * contexts
 * 
 * @author Josh Attenberg
 */
public class NumericCSVtoLabeledVectorPipe extends
		AbstractPipe<String, LinearVector> {

	private static final long serialVersionUID = 560043518941752078L;
	private Map<String, String> labelMap = null;
	private int labelColumn = -1;
	private int nameColumn = -1;
	private String delimiter = ",";

	/**
	 * Class constructor
	 */
	public NumericCSVtoLabeledVectorPipe() {
		super();
	}

	/**
	 * Class constructor specifying number of nameColumn to create
	 * 
	 * @param nameColumn
	 *            number of nameColumn
	 */
	public NumericCSVtoLabeledVectorPipe(int nameColumn) {
		this();
		this.nameColumn = nameColumn;
	}

	/**
	 * Class constructor specifying delimiter to create
	 * 
	 * @param delimiter
	 *            delimiter
	 */
	public NumericCSVtoLabeledVectorPipe(String delimiter) {
		this();
		this.delimiter = delimiter;
	}

	/**
	 * Class constructor specifying delimiter and number of nameColumn to create
	 * 
	 * @param delimiter
	 *            delimiter
	 * @param nameColumn
	 *            umber of nameColumn
	 */
	public NumericCSVtoLabeledVectorPipe(String delimiter, int nameColumn) {
		this(nameColumn);
		this.delimiter = delimiter;
	}

	/**
	 * Class constructor specifying number of labelColumn and number of
	 * nameColumn to create
	 * 
	 * @param nameColumn
	 *            number of nameColumn
	 * @param labelColumn
	 *            number of labelColumn
	 */
	public NumericCSVtoLabeledVectorPipe(int nameColumn, int labelColumn) {
		this(nameColumn);
		this.labelColumn = labelColumn;
	}

	/**
	 * Class constructor specifying number of labelColumn, number of nameColumn
	 * and delimiter to create
	 * 
	 * @param delimiter
	 *            delimiter
	 * @param nameColumn
	 *            number of nameColumn
	 * @param labelColumn
	 *            number of labelColumn
	 */
	public NumericCSVtoLabeledVectorPipe(String delimiter, int nameColumn,
			int labelColumn) {
		this(nameColumn, labelColumn);
		this.delimiter = delimiter;
	}

	/**
	 * Class constructor specifying number of labelColumn, and the labelMap which 
	 * specifies multiple labels to create
	 * 
	 * @param labelColumn
	 *            number of labelColumn
	 * @param labelMap
	 *            multiple label
	 */
	public NumericCSVtoLabeledVectorPipe(int labelColumn,
			Map<String, String> labelMap) {
		super();
		this.labelColumn = labelColumn;
		if (labelColumn > 0 && (labelMap == null || labelMap.isEmpty()))
			throw new IllegalArgumentException(
					"with non-zero label column, labelMap must be non-null ");
		this.labelMap = labelMap;
	}

	/**
	 * Class constructor specifying number of nameColumn, number of labelColumn
	 * and multiple label to create
	 * 
	 * @param nameColumn
	 *            number of nameColumn
	 * @param labelColumn
	 *            number of labelColumn
	 * @param labelMap
	 *            multiple label
	 */
	public NumericCSVtoLabeledVectorPipe(int nameColumn, int labelColumn,
			Map<String, String> labelMap) {
		this(labelColumn, labelMap);
		this.nameColumn = nameColumn;
	}

	/**
	 * Class constructor specifying delimiter,number of labelColumn and multiple
	 * label to create
	 * 
	 * @param delimiter
	 *            delimiter
	 * @param labelColumn
	 *            number of labelColumn
	 * @param labelMap
	 *            multiple label
	 */
	public NumericCSVtoLabeledVectorPipe(String delimiter, int labelColumn,
			Map<String, String> labelMap) {
		this(labelColumn, labelMap);
		this.delimiter = delimiter;
	}

	/**
	 * Class constructor specifying delimiter,number of nameColumn,number of
	 * labelColumn and multiple label to create
	 * 
	 * @param delimiter
	 *            delimiter
	 * @param nameColumnm
	 *            number of nameColumn
	 * @param labelColumn
	 *            number of labelColumn
	 * @param labelMap
	 *            multiple label
	 */
	public NumericCSVtoLabeledVectorPipe(String delimiter, int nameColumnm,
			int labelColumn, Map<String, String> labelMap) {
		this(delimiter, labelColumn, labelMap);
		this.nameColumn = nameColumnm;
	}

	/**
	 * The method returns the class's Type "NumericCSVtoLabeledVectorPipe"
	 * 
	 * @return Type
	 */
	@Override
	public Type getType() {
		return new TypeToken<NumericCSVtoLabeledVectorPipe>() {
		}.getType();
	}

	@Override
	protected Context<LinearVector> operate(Context<String> context) {
		String line = context.getData();
		String[] parts = line.split(delimiter);

		if (labelColumn >= 0 && labelColumn >= parts.length)
			throw new IllegalStateException(
					"attempting to use a label column at position: "
							+ labelColumn + " with size: " + parts.length);

		if (nameColumn >= 0 && nameColumn >= parts.length)
			throw new IllegalStateException(
					"attempting to use a name column at position: "
							+ nameColumn + " with size: " + parts.length);

		String name = null;
		if (nameColumn >= 0)
			name = parts[nameColumn];
		else
			name = context.getId();

		String label = null;
		if (labelColumn >= 0)
			label = (null != labelMap && labelMap
					.containsKey(parts[labelColumn])) ? labelMap
					.get(parts[labelColumn]) : parts[labelColumn];
		else
			label = context.getLabel();

		/**
		 * Problem here: if the label or name columns aren't at the end of the
		 * input csv then the size on the output vector will be less than the
		 * indices of the input features there are two possible fixes: 1) keep
		 * the index of features from the CSV in the input vectors and have some
		 * "blank" features in the vector 2) "shift right" on the input
		 * features, loosing the mapping from input column ids to output vector
		 * dimensions, but eliminating any useless dimensions.
		 * 
		 * here, we're choosing approach 1.
		 */
		LinearVector vect = LinearVectorFactory.getDenseVector(parts.length
				- (nameColumn >= 0 ? 1 : 0) - (labelColumn >= 0 ? 1 : 0));
		int ct = 0;
		for (int i = 0; i < parts.length; i++) {			
			if (i == labelColumn || i == nameColumn) {
				continue;
			}
			vect.resetValue(ct++, Double.parseDouble(parts[i].trim()));
		}
		
		Context<LinearVector> out = Context.createContext(context, vect);
		out.setId(name);
		out.setLabel(label);
		return out;
	}
}
