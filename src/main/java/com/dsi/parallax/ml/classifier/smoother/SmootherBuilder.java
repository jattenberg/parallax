package com.dsi.parallax.ml.classifier.smoother;

import com.dsi.parallax.ml.util.bounds.GreaterThanValueBound;
import com.dsi.parallax.ml.util.option.*;
import com.dsi.parallax.ml.util.pair.PrimitivePair;

import java.util.Collection;

import static com.google.common.base.Preconditions.checkArgument;

public abstract class SmootherBuilder<R extends Smoother<R>, B extends SmootherBuilder<R, B>>
		extends AbstractConfigurable<B> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4973573668568000527L;

	protected final SmootherType type;
	protected B smootherBuilder;

	protected SmootherBuilder(SmootherType type) {
		this.type = type;
		smootherBuilder = getThis();
	}

	protected abstract B getThis();

	protected abstract R build();

	public R train(Collection<PrimitivePair> input) {
		R smoother = build();
		smoother.train(input);
		return smoother;
	}

	public static class BinningSmootherBuilder extends
			SmootherBuilder<BinningSmoother, BinningSmootherBuilder> {

		private static final long serialVersionUID = -6994635274519452681L;
		public static final BinningSmootherOptions options = new BinningSmootherOptions();

		private int bins = 25;

		public BinningSmootherBuilder() {
			super(SmootherType.BINNING);
		}

		@Override
		public Configuration<BinningSmootherBuilder> getConfiguration() {
			Configuration<BinningSmootherBuilder> conf = new Configuration<SmootherBuilder.BinningSmootherBuilder>(
					options);
			populateConfiguration(conf);
			return conf;
		}

		@Override
		public Configuration<BinningSmootherBuilder> populateConfiguration(
				Configuration<BinningSmootherBuilder> conf) {
			conf.addIntegerValueOnShortName("b", bins);
			return conf;
		}

		@Override
		public void configure(
				Configuration<BinningSmootherBuilder> configuration) {
			setBins(configuration.integerOptionFromShortName("b"));
		}

		@Override
		protected BinningSmootherBuilder getThis() {
			return this;
		}

		@Override
		protected BinningSmoother build() {
			return new BinningSmoother(bins);
		}

		public BinningSmootherBuilder setBins(int bins) {
			checkArgument(bins > 0, "bins must be positive, given: %s", bins);
			this.bins = bins;
			return smootherBuilder;
		}

		public int getBins() {
			return this.bins;
		}

		protected static class BinningSmootherOptions extends
				OptionSet<BinningSmootherBuilder> {
			{
				addOption(new IntegerOption(
						"b",
						"bins",
						"The number of classifier score bins to use for binning smoother",
						25, false, new GreaterThanValueBound(0)));
			}
		}
	}

	public static class UpdateableLogisticRegressionSmootherBuilder
			extends
			SmootherBuilder<UpdateableLogisticRegressionSmoother, UpdateableLogisticRegressionSmootherBuilder> {

		private static final long serialVersionUID = 6647699485931111002L;
		public static final UpdateableLogisticRegressionSmootherOptions options = new UpdateableLogisticRegressionSmootherOptions();
		private int passes = 30;
		private double weight = 0.1d;

		public UpdateableLogisticRegressionSmootherBuilder() {
			super(SmootherType.UPDATEABLEPLATT);
		}

		@Override
		public Configuration<UpdateableLogisticRegressionSmootherBuilder> getConfiguration() {
			Configuration<UpdateableLogisticRegressionSmootherBuilder> configuration = new Configuration<SmootherBuilder.UpdateableLogisticRegressionSmootherBuilder>(
					options);
			populateConfiguration(configuration);
			return configuration;
		}

		@Override
		public Configuration<UpdateableLogisticRegressionSmootherBuilder> populateConfiguration(
				Configuration<UpdateableLogisticRegressionSmootherBuilder> conf) {
			conf.addIntegerValueOnShortName("p", passes);
			conf.addFloatValueOnShortName("w", weight);
			return conf;
		}

		@Override
		public void configure(
				Configuration<UpdateableLogisticRegressionSmootherBuilder> configuration) {
			setPasses(configuration.integerOptionFromShortName("p"));
			setWeight(configuration.floatOptionFromShortName("w"));

		}

		@Override
		protected UpdateableLogisticRegressionSmootherBuilder getThis() {
			return this;
		}

		@Override
		protected UpdateableLogisticRegressionSmoother build() {
			UpdateableLogisticRegressionSmoother smoother = new UpdateableLogisticRegressionSmoother()
					.setPasses(passes).setWeight(weight);
			return smoother;
		}

		public int getPasses() {
			return passes;
		}

		public double getWeight() {
			return weight;
		}

		public UpdateableLogisticRegressionSmootherBuilder setPasses(
				int passes) {
			checkArgument(passes > 0, "passes must be positive, given: %s",
					passes);
			this.passes = passes;
			return smootherBuilder;
		}

		public UpdateableLogisticRegressionSmootherBuilder setWeight(
				double weight) {
			checkArgument(weight > 0, "weight must be positive, given: %s",
					weight);
			this.weight = weight;
			return smootherBuilder;
		}

		protected static class UpdateableLogisticRegressionSmootherOptions
				extends
				OptionSet<UpdateableLogisticRegressionSmootherBuilder> {
			{
				addOption(new IntegerOption(
						"p",
						"passes",
						"The number of passes used for sgd training over all the training instances",
						25, false, new GreaterThanValueBound(0)));
				addOption(new FloatOption(
						"w",
						"weight",
						"initial weight for sgd steps. weight is decresed by 0.1 each pass",
						0.1, false, new GreaterThanValueBound(0)));
			}
		}
	}

	public static class LogisticRegressionSmootherBuilder
			extends
			SmootherBuilder<LogisticRegressionSmoother, LogisticRegressionSmootherBuilder> {

		private static final long serialVersionUID = -7118115876801986526L;
		public static final LogisticRegressionSmootherOptions options = new LogisticRegressionSmootherOptions();

		private int passes = 10;

		public LogisticRegressionSmootherBuilder() {
			super(SmootherType.PLATT);
		}

		@Override
		public Configuration<LogisticRegressionSmootherBuilder> getConfiguration() {
			Configuration<LogisticRegressionSmootherBuilder> conf = new Configuration<LogisticRegressionSmootherBuilder>(
					options);
			populateConfiguration(conf);
			return conf;
		}

		@Override
		public Configuration<LogisticRegressionSmootherBuilder> populateConfiguration(
				Configuration<LogisticRegressionSmootherBuilder> conf) {
			conf.addIntegerValueOnShortName("p", passes);
			return conf;
		}

		@Override
		public void configure(
				Configuration<LogisticRegressionSmootherBuilder> configuration) {
			setPasses(configuration.integerOptionFromShortName("p"));

		}

		@Override
		protected LogisticRegressionSmootherBuilder getThis() {
			return this;
		}

		@Override
		protected LogisticRegressionSmoother build() {
			LogisticRegressionSmoother smoother = new LogisticRegressionSmoother();
			smoother.setPasses(passes);
			return smoother;
		}

		public LogisticRegressionSmootherBuilder setPasses(int passes) {
			checkArgument(passes > 0, "passes must be postitive, given: %s",
					passes);
			this.passes = passes;
			return smootherBuilder;
		}

		public int getPasses() {
			return passes;
		}

		protected static class LogisticRegressionSmootherOptions extends
				OptionSet<LogisticRegressionSmootherBuilder> {
			{
				addOption(new IntegerOption(
						"p",
						"passes",
						"The number of passes used for sgd training over all the training instances",
						25, false, new GreaterThanValueBound(0)));

			}
		}
	}

	public static class IsotonicRegressionBuilder extends
			SmootherBuilder<IsotonicSmoother, IsotonicRegressionBuilder> {

		private static final long serialVersionUID = -3072642670037900193L;
		public static final IsotonicRegressionOptions options = new IsotonicRegressionOptions();

		public IsotonicRegressionBuilder() {
			super(SmootherType.ISOTONIC);
		}

		@Override
		public Configuration<IsotonicRegressionBuilder> getConfiguration() {
			Configuration<IsotonicRegressionBuilder> conf = new Configuration<SmootherBuilder.IsotonicRegressionBuilder>(
					options);
			populateConfiguration(conf);
			return conf;
		}

		@Override
		public Configuration<IsotonicRegressionBuilder> populateConfiguration(
				Configuration<IsotonicRegressionBuilder> conf) {
			// nothing to populate
			return conf;
		}

		@Override
		public void configure(
				Configuration<IsotonicRegressionBuilder> configuration) {
			// nothing to configure.
		}

		@Override
		protected IsotonicRegressionBuilder getThis() {
			return this;
		}

		@Override
		protected IsotonicSmoother build() {
			return new IsotonicSmoother();
		}

		public static class IsotonicRegressionOptions extends
				OptionSet<IsotonicRegressionBuilder> {

		}
	}

	public static class LogitSmootherBuilder extends
			SmootherBuilder<LogitSmoother, LogitSmootherBuilder> {

		private static final long serialVersionUID = -3072642670037900193L;
		public static final LogitSmootherOptions options = new LogitSmootherOptions();

		public LogitSmootherBuilder() {
			super(SmootherType.LOGIT);
		}

		@Override
		public Configuration<LogitSmootherBuilder> getConfiguration() {
			Configuration<LogitSmootherBuilder> conf = new Configuration<SmootherBuilder.LogitSmootherBuilder>(
					options);
			populateConfiguration(conf);
			return conf;
		}

		@Override
		public Configuration<LogitSmootherBuilder> populateConfiguration(
				Configuration<LogitSmootherBuilder> conf) {
			// nothing to populate
			return conf;
		}

		@Override
		public void configure(
				Configuration<LogitSmootherBuilder> configuration) {
			// nothing to configure.
		}

		@Override
		protected LogitSmootherBuilder getThis() {
			return this;
		}

		@Override
		protected LogitSmoother build() {
			return new LogitSmoother();
		}

		public static class LogitSmootherOptions extends
				OptionSet<LogitSmootherBuilder> {

		}
	}

	public static class NullSmootherBuilder extends
			SmootherBuilder<NullSmoother, NullSmootherBuilder> {

		private static final long serialVersionUID = -3072642670037900193L;
		public static final NullSmootherOptions options = new NullSmootherOptions();

		public NullSmootherBuilder() {
			super(SmootherType.NONE);
		}

		@Override
		public Configuration<NullSmootherBuilder> getConfiguration() {
			Configuration<NullSmootherBuilder> conf = new Configuration<SmootherBuilder.NullSmootherBuilder>(
					options);
			populateConfiguration(conf);
			return conf;
		}

		@Override
		public Configuration<NullSmootherBuilder> populateConfiguration(
				Configuration<NullSmootherBuilder> conf) {
			// nothing to populate
			return conf;
		}

		@Override
		public void configure(
				Configuration<NullSmootherBuilder> configuration) {
			// nothing to configure.
		}

		@Override
		protected NullSmootherBuilder getThis() {
			return this;
		}

		@Override
		protected NullSmoother build() {
			return new NullSmoother();
		}

		public static class NullSmootherOptions extends
				OptionSet<NullSmootherBuilder> {

		}
	}

	public static abstract class AbstractKNNSmootherBuilder<R extends AbstractKNNSmoother<R>, B extends AbstractKNNSmootherBuilder<R, B>>
			extends SmootherBuilder<R, B> {

		private static final long serialVersionUID = 608044841350804927L;
		protected int k = 10;

		protected AbstractKNNSmootherBuilder(SmootherType type) {
			super(type);
		}

		@Override
		public Configuration<B> populateConfiguration(Configuration<B> conf) {
			conf.addIntegerValueOnShortName("k", k);
			return conf;
		}

		@Override
		public void configure(Configuration<B> configuration) {
			setK(configuration.integerOptionFromShortName("k"));
		}

		public B setK(int k) {
			checkArgument(k > 0, "k must be positive, given: %s", k);
			this.k = k;
			return smootherBuilder;
		}

		public int getK() {
			return k;
		}

		protected static abstract class AbstractKNNSmootherOptions<R extends AbstractKNNSmoother<R>, B extends AbstractKNNSmootherBuilder<R, B>>
				extends OptionSet<B> {
			{
				addOption(new IntegerOption("k", "k",
						"k used for k nearest neighbor algorithms", 10, true,
						new GreaterThanValueBound(0)));
			}
		}
	}

	public static class KNNSmootherBuilder
			extends
			AbstractKNNSmootherBuilder<KNNSmoother, KNNSmootherBuilder> {

		private static final long serialVersionUID = 3707320554524247023L;
		public static final KNNSmootherOptions options = new KNNSmootherOptions();

		public KNNSmootherBuilder() {
			super(SmootherType.KNN);
		}

		@Override
		public Configuration<KNNSmootherBuilder> getConfiguration() {
			Configuration<KNNSmootherBuilder> config = new Configuration<KNNSmootherBuilder>(
					options);
			populateConfiguration(config);
			return config;
		}

		@Override
		protected KNNSmootherBuilder getThis() {
			return this;
		}

		@Override
		protected KNNSmoother build() {
			KNNSmoother smoother = new KNNSmoother();
			smoother.setK(k);
			return smoother;
		}

		public static class KNNSmootherOptions
				extends
				AbstractKNNSmootherOptions<KNNSmoother, KNNSmootherBuilder> {
		}
	}

	public static class LocalLogisticRegressionSmootherBuilder
			extends
			AbstractKNNSmootherBuilder<LocalLogisticRegressionSmoother, LocalLogisticRegressionSmootherBuilder> {

		private static final long serialVersionUID = 8799355863896462391L;
		public static final LocalLogisticRegressionSmootherOptions options = new LocalLogisticRegressionSmootherOptions();
		private int passes = 10;

		public LocalLogisticRegressionSmootherBuilder() {
			super(SmootherType.LOCALLR);
		}

		@Override
		public Configuration<LocalLogisticRegressionSmootherBuilder> getConfiguration() {
			Configuration<LocalLogisticRegressionSmootherBuilder> conf = new Configuration<LocalLogisticRegressionSmootherBuilder>(
					options);
			populateConfiguration(conf);
			return conf;
		}

		@Override
		protected LocalLogisticRegressionSmootherBuilder getThis() {
			return this;
		}

		@Override
		protected LocalLogisticRegressionSmoother build() {
			LocalLogisticRegressionSmoother smoother = new LocalLogisticRegressionSmoother();
			smoother.setK(k).setPasses(passes);
			return smoother;
		}

		@Override
		public Configuration<LocalLogisticRegressionSmootherBuilder> populateConfiguration(
				Configuration<LocalLogisticRegressionSmootherBuilder> conf) {
			super.populateConfiguration(conf);
			conf.addIntegerValueOnShortName("p", passes);
			return conf;
		}

		public LocalLogisticRegressionSmootherBuilder setPasses(int passes) {
			checkArgument(passes > 0, "passes must be positive, given: %s",
					passes);
			this.passes = passes;
			return smootherBuilder;
		}

		public int getPasses() {
			return passes;
		}

		public static class LocalLogisticRegressionSmootherOptions
				extends
				AbstractKNNSmootherOptions<LocalLogisticRegressionSmoother, LocalLogisticRegressionSmootherBuilder> {
			{
				addOption(new IntegerOption("p", "passes",
						"number of passes for sgd-like optimization", 10,
						false, new GreaterThanValueBound(0)));
			}
		}
	}

}
