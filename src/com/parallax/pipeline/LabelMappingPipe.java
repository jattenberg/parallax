package com.parallax.pipeline;

import java.lang.reflect.Type;
import java.util.Map;

import com.google.gson.reflect.TypeToken;
import com.parallax.ml.vector.LinearVector;
import com.parallax.pipeline.AbstractPipe;
import com.parallax.pipeline.Context;


/**
 * LabelMappingPipe converts arbitrary labels to binary 0/1 labels using the
 * given labelmap
 *
 * @author Sumit Chopra
 */
public class LabelMappingPipe extends AbstractPipe<LinearVector, LinearVector> {

	/**
	 * Static serial id for the class
	 */
	private static final long serialVersionUID = 7735757748842660347L;

	/**
	 * the label map to be used for converting the labels
	 */
	private Map<String, String> labelMap = null; 
	
    /**
     * Class constructor specifying the LabelMap
     * @param lblmap
     * 				The label map to be used for mapping arbitrary labels 
     * 				into binary (0/1) labels
     */
	public LabelMappingPipe(Map<String, String> lblmap) {
		super();
		this.labelMap = lblmap;
	}
    
    /**
     * The method returns the class's Type "LabelMappingPipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<LabelMappingPipe>() {
		}.getType();
	}

	@Override
	protected Context<LinearVector> operate(Context<LinearVector> context) {
		// get the fields from the incoming context
		LinearVector oldData = context.getData();
		String oldLabel = context.getLabel();
		String oldId = context.getId();
		
		// map the old label to the new label using the labelMap
		String newLabel = labelMap.get(oldLabel);
		
		// return a new context with the newLabel and old data and Id
		return Context.createContext(context, oldData, oldId, newLabel);
	}
}
