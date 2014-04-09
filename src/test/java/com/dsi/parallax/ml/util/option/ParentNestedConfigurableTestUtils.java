package com.dsi.parallax.ml.util.option;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertArrayEquals;

import java.util.EnumSet;

import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.NestedConfigurable;
import com.dsi.parallax.ml.util.option.NestedConfigurableType;
import com.dsi.parallax.ml.util.option.ParentNestedConfigurable;
import com.dsi.parallax.ml.util.option.ParentNestedConfigurableOptionSet;

public class ParentNestedConfigurableTestUtils {

	public static <T extends Enum<T> & NestedConfigurableType<C, T>, C extends NestedConfigurable<C, T>, P extends ParentNestedConfigurable<T, C, P>> void testMatchesConfig(
			Class<T> nestedTypeClass, P parentConfigurable) {

		for (T type : EnumSet.allOf(nestedTypeClass)) {
			parentConfigurable.setConfigurableType(type);
			assertEquals(type, parentConfigurable.getCurrentType());
			assertEquals(type, parentConfigurable.configurableForType(type)
					.correspondingType());
			assertEquals(type, parentConfigurable.currentNestedConfigurable()
					.correspondingType());
		}
	}

	public static <T extends Enum<T> & NestedConfigurableType<C, T>, C extends NestedConfigurable<C, T>, P extends ParentNestedConfigurable<T, C, P>> void testBuildsCorrectType(
			Class<T> nestedTypeClass, P parentConfigurable) {

		Configuration<P> config = parentConfigurable.getConfiguration();

		for (T type : EnumSet.allOf(nestedTypeClass)) {
			C trueConfigurable = type.getConfigurable();
			config.addEnumValueOnShortName(
					ParentNestedConfigurableOptionSet.TYPESHORT, type);
			config.addConfigurableValueOnShortName(
					ParentNestedConfigurableOptionSet.CONFIGSHORT,
					type.getDefaultConfiguration());

			parentConfigurable.configure(config);
			C testConfigurable = parentConfigurable.currentNestedConfigurable();

			assertEquals(trueConfigurable.getClass().getCanonicalName(),
					testConfigurable.getClass().getCanonicalName());
		}
	}

	public static <T extends Enum<T> & NestedConfigurableType<C, T>, C extends NestedConfigurable<C, T>, P extends ParentNestedConfigurable<T, C, P>> void testNestedConfigurations(
			Class<T> nestedTypeClass, P parentConfigurable) {

		Configuration<P> config = parentConfigurable.getConfiguration();

		for (T type : EnumSet.allOf(nestedTypeClass)) {
			Configuration<C> trueConfigurable = type.getConfigurable()
					.getConfiguration();
			config.addEnumValueOnShortName(
					ParentNestedConfigurableOptionSet.TYPESHORT, type);
			config.addConfigurableValueOnShortName(
					ParentNestedConfigurableOptionSet.CONFIGSHORT,
					type.getDefaultConfiguration());

			parentConfigurable.configure(config);
			Configuration<C> testConfigurable = parentConfigurable
					.currentNestedConfigurable().getConfiguration();

			assertArrayEquals(trueConfigurable.getArgumentsFromOpts(),
					testConfigurable.getArgumentsFromOpts());
		}
	}
}
