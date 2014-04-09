package com.dsi.parallax.ml.util.option;

/**
 * option where several separate configurable types may be realized- eg, there
 * are several different annealing schedules, it only makes sense to take one,
 * but each option has diffrernt configurable values.
 * 
 * in this case, each of the T enum defines a configurable
 * 
 * @author jattenberg
 * 
 */
public class NestedConfigurableOption<T extends Enum<T> & NestedConfigurableType<C, T>, C extends NestedConfigurable<C, T>, P extends ParentNestedConfigurable<T, C, P>>
		extends ConfigurableOption<P> {

	private static final long serialVersionUID = 1641996640930522988L;

	protected final ParentNestedConfigurableOptionSet<T, C, P> options;

	public NestedConfigurableOption(String shortName, String longName,
			boolean optimizable, String desc,
			ParentNestedConfigurableOptionSet<T, C, P> options) {
		super(shortName, longName, optimizable, desc, new Configuration<P>(
				options));
		this.options = options;
	}

	@Override
	public OptionType getType() {
		return OptionType.NESTEDCONFIGURABLE;
	}

	@Override
	public Configuration<P> getDefaultConfig() {
		return new NestedConfiguration<T, C, P>(options);
	}
}
