package com.dsi.parallax.ml.util.option;

public interface Setter<T> {

	public void set(T value);
	
	public interface IntegerSetter extends Setter<Integer> {
		
	}
	
	public interface FloatSetter extends Setter<Double> {
		
	}
	
	public interface BooleanSetter extends Setter<Boolean> {
		
	}
	
	public interface EnumSetter extends Setter<Enum<?>> {
		
	}
	
	public interface ConfigurationSetter extends Setter<Configuration<?>> {
		
	}
}
