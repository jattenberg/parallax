package com.parallax.ml.util.option;

public abstract class NestedConfigurable<C extends NestedConfigurable<C, T>, T extends Enum<T> & NestedConfigurableType<C, T>>
		extends AbstractConfigurable<C> {

	private static final long serialVersionUID = -7421016213594575823L;

	
	public abstract Configuration<C> defaultConfiguration();
	public abstract T correspondingType();
		
}