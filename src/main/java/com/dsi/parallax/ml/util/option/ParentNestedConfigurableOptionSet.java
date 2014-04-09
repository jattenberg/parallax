package com.dsi.parallax.ml.util.option;

import java.util.EnumSet;

public class ParentNestedConfigurableOptionSet<T extends Enum<T> & NestedConfigurableType<C, T>, C extends NestedConfigurable<C, T>, P extends ParentNestedConfigurable<T, C, P>>
		extends OptionSet<P> {

	public static final String TYPESHORT = "T";
	public static final String TYPELONG = "type";
	private static final String DESCRIPTION_PREFIX = "The NestedConfigurableType Enum value enumerating the types of nested configurables that can be used. options: ";

	public static final String CONFIGSHORT = "C";
	public static final String CONFIGLONG = "configuration";
	private static final String CONFIGURATION_DESCRIPTION = "The Configurable option corresponding to the Nested Configurable";

	protected ParentNestedConfigurableOptionSet(boolean optimizable,
			Class<T> enumclass, T defaulte) {
		super();
		addOption(new EnumOption<T>(TYPESHORT, TYPELONG, optimizable,
				DESCRIPTION_PREFIX + EnumSet.allOf(enumclass), enumclass,
				defaulte));

		addOption(new ConfigurableOption<C>(CONFIGSHORT, CONFIGLONG,
				optimizable, CONFIGURATION_DESCRIPTION,
				defaulte.getDefaultConfiguration()));
	}

	public T defaultType() {
		@SuppressWarnings("unchecked")
		EnumOption<T> typeOption = (EnumOption<T>) shortNameToOptions
				.get(TYPESHORT);
		return typeOption.DEFAULTE;
	}

	public Configuration<C> defaultNestedConfiguration() {
		T type = defaultType();
		return type.getDefaultConfiguration();
	}
}
