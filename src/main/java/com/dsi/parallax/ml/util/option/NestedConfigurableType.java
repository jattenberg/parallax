package com.dsi.parallax.ml.util.option;

/**
 * TODO: ensrue this extends an Enum
 * @author jattenberg
 *
 * @param <C>
 * @param <T>
 */
public interface NestedConfigurableType<C extends NestedConfigurable<C, T>, T extends Enum<T> & NestedConfigurableType<C, T>> {
	
	public abstract C getConfigurable();

	public abstract Configuration<C> getDefaultConfiguration();
}