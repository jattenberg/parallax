/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline;
/**
 * carries info through a pipeline
 * @author jattenberg
 */
public class Context<T> {

	String id;
    String label;
    T payload;
    
    public Context(T o) {
        id = null;
        label = null;
        this.payload = o;
    }
    
    public Context(String id, String label, T o) {
        this.id = id;
        this.label = label;
        this.payload = o;
    }
    
    public Context(Context<T> context) {
        this.id = context.id;
        this.label = context.label;
        this.payload = context.payload;
    }
    
    public T getData() {
        return payload;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public void SetData(T o) {
        this.payload = o;
    }
    
    @SuppressWarnings("unchecked")
    public static <T> T unsecureCast(Object o) {
		return (T) o;
    }
	
	public static <T> Context<T> createContext(T data) {
		return new Context<T>(data);
	}
	
	public static <T> Context<T> createContext(String id, String label, T o) {
		return new Context<T>(id, label, o);
	}

	public static <T> Context<T> createContext(Context<?> context, T payload) {
		Context<T> out = unsecureCast(context);
		out.payload = payload;
		return out;
	}
	
	public static <T> Context<T> createContext(Context<?> context, T payload, String id, String label) {
		Context<T> out = unsecureCast(context);
		out.payload = payload;
		out.id = id;
		out.label = label;
		return out;
	}
	
	@Override
	public String toString() {
		return "("+label+"): " + id + " " + payload;
	}
}
