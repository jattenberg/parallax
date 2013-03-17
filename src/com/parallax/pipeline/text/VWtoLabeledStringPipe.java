/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.text;

import java.lang.reflect.Type;
import java.util.Map;

import com.google.gson.reflect.TypeToken;
import com.parallax.ml.util.VW;
import com.parallax.pipeline.AbstractPipe;
import com.parallax.pipeline.Context;

/**
 * very primitive class for getting contents of VW object into context 
 *
 * @author jattenberg
 */
public class VWtoLabeledStringPipe extends AbstractPipe<VW,String> {

    private static final long serialVersionUID = 143740037785791670L;
    private static final String SEP = " ";
	StringBuilder buff;

    /**
     * Class constructor
     */
    public VWtoLabeledStringPipe() {
    	super();
    	buff = new StringBuilder();
    }

    /**
     * The method returns the class's Type "VWtoLabeledStringPipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<VWtoLabeledStringPipe>(){}.getType();
	}

    private String bodyFromVWData(Map<String, String> namespaceData) {
        buff.setLength(0);
        
        boolean first = true;
        for(String key : namespaceData.keySet()) {
            String value = namespaceData.get(key);
            buff.append(first ? "" : SEP).append(value);
            first = false;
        }
        return buff.toString();
    }
    
	@Override
	protected Context<String> operate(Context<VW> context) {
        VW vw = context.getData();
        Map<String, String> namespaceData = vw.getNamespaceData();
        String body = bodyFromVWData(namespaceData);
        Context<String> out = Context.createContext(context, body);
		out.setId(vw.getID());
		out.setLabel(vw.getCategory());
		return out;
	}
}
