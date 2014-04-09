package com.dsi.parallax.optimization.stochastic.anneal;

import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.NestedConfiguration;
import com.dsi.parallax.ml.util.option.ParentNestedConfigurable;
import com.dsi.parallax.ml.util.option.ParentNestedConfigurableOptionSet;

public class AnnealingScheduleConfigurableBuilder
		extends
		ParentNestedConfigurable<AnnealingScheduleType, AnnealingScheduleBuilder, AnnealingScheduleConfigurableBuilder> {

	private static final long serialVersionUID = -8953188482690371956L;

	public static final AnnealingScheduleConfigurableBuilderOptions options = new AnnealingScheduleConfigurableBuilderOptions(
			true);

	public AnnealingScheduleConfigurableBuilder() {
		this(new Configuration<AnnealingScheduleConfigurableBuilder>(options));
	}

	public AnnealingScheduleConfigurableBuilder(
			Configuration<AnnealingScheduleConfigurableBuilder> configuration) {
		super();
		configure(configuration);
	}

	@Override
	public NestedConfiguration<AnnealingScheduleType, AnnealingScheduleBuilder, AnnealingScheduleConfigurableBuilder> getConfiguration() {
		NestedConfiguration<AnnealingScheduleType, AnnealingScheduleBuilder, AnnealingScheduleConfigurableBuilder> configuration = NestedConfiguration
				.buildNestedConfiguration(options);
		populateConfiguration(configuration);
		return configuration;
	}

	@Override
	public Configuration<AnnealingScheduleConfigurableBuilder> populateConfiguration(
			Configuration<AnnealingScheduleConfigurableBuilder> conf) {
		conf.addEnumValueOnShortName(TYPESHORT, currentType);
		conf.addConfigurableValueOnShortName(CONFIGSHORT,
				currentConfigurable.getConfiguration());
		return conf;
	}

	@SuppressWarnings("unchecked")
	@Override
	public void configure(
			Configuration<AnnealingScheduleConfigurableBuilder> configuration) {
		setConfigurableType((AnnealingScheduleType) configuration
				.enumFromShortName(ParentNestedConfigurableOptionSet.TYPESHORT));
		setInternalConfiguration((Configuration<AnnealingScheduleBuilder>) configuration
				.configurationFromShortName(ParentNestedConfigurableOptionSet.CONFIGSHORT));
	}

	@Override
	public AnnealingScheduleType[] childTypes() {
		return AnnealingScheduleType.values();
	}

	@Override
	public AnnealingScheduleBuilder configurableForType(
			AnnealingScheduleType type) {
		return type.getConfigurable();
	}

	@Override
	public Configuration<AnnealingScheduleBuilder> currentNestedConfiguration() {
		return currentConfigurable.getConfiguration();
	}

	public AnnealingSchedule build() {
		return currentConfigurable.build();
	}

	public AnnealingScheduleConfigurableBuilder setInternalConfiguration(
			Configuration<AnnealingScheduleBuilder> configuration) {
		currentConfigurable.configure(configuration);
		return this;
	}

	@Override
	public AnnealingScheduleBuilder currentNestedConfigurable() {
		return currentConfigurable;
	}

	public static class AnnealingScheduleConfigurableBuilderOptions
			extends
			ParentNestedConfigurableOptionSet<AnnealingScheduleType, AnnealingScheduleBuilder, AnnealingScheduleConfigurableBuilder> {

		public AnnealingScheduleConfigurableBuilderOptions(boolean optimizable) {
			super(optimizable, AnnealingScheduleType.class,
					AnnealingScheduleType.INVERSE);
		}

	}

	@Override
	protected AnnealingScheduleConfigurableBuilder getParentConfigurable() {
		return this;
	}

	public static AnnealingScheduleConfigurableBuilder configureForConstantRate(
			double rate) {
		AnnealingScheduleConfigurableBuilder ascb = new AnnealingScheduleConfigurableBuilder();
		ascb.setConfigurableType(AnnealingScheduleType.CONSTANT);
		Configuration<AnnealingScheduleBuilder> internalConfig = new Configuration<AnnealingScheduleBuilder>(
				ConstantAnnealingScheduleBuilder.options);
		internalConfig.addFloatValueOnShortName("r", rate);
		ascb.setInternalConfiguration(internalConfig);
		return ascb;
	}

	public static AnnealingScheduleConfigurableBuilder configureForInverseDecay(
			double rate, double decay) {
		AnnealingScheduleConfigurableBuilder ascb = new AnnealingScheduleConfigurableBuilder();
		ascb.setConfigurableType(AnnealingScheduleType.INVERSE);
		Configuration<AnnealingScheduleBuilder> internalConfig = new Configuration<AnnealingScheduleBuilder>(
				InverseDecayAnnealingScheduleBuilder.options);
		internalConfig.addFloatValueOnShortName("r", rate);
		internalConfig.addFloatValueOnShortName("d", decay);
		ascb.setInternalConfiguration(internalConfig);
		return ascb;
	}

	public static AnnealingScheduleConfigurableBuilder configureForExponentialDecay(
			double rate, double base) {
		AnnealingScheduleConfigurableBuilder ascb = new AnnealingScheduleConfigurableBuilder();
		ascb.setConfigurableType(AnnealingScheduleType.EXPONENTIAL);
		Configuration<AnnealingScheduleBuilder> internalConfig = new Configuration<AnnealingScheduleBuilder>(
				ExponentialAnnealingScheduleBuilder.options);
		internalConfig.addFloatValueOnShortName("r", rate);
		internalConfig.addFloatValueOnShortName("b", base);
		ascb.setInternalConfiguration(internalConfig);
		return ascb;
	}

}
