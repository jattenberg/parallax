package com.parallax.ml.metaoptimize;

import static com.google.common.base.Preconditions.checkArgument;

import com.parallax.ml.classifier.Classifier;
import com.parallax.ml.classifier.ClassifierBuilder;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.objective.ObjectiveScorer;
import com.parallax.ml.target.BinaryClassificationTarget;
import com.parallax.ml.util.option.BooleanOption;
import com.parallax.ml.util.option.Configurable;
import com.parallax.ml.util.option.ConfigurableOption;
import com.parallax.ml.util.option.Configuration;
import com.parallax.ml.util.option.EnumOption;
import com.parallax.ml.util.option.FloatOption;
import com.parallax.ml.util.option.IntegerOption;
import com.parallax.ml.util.option.Option;

// TODO: Auto-generated Javadoc
/**
 * The Class OptionOptimizer.
 *
 * @param <C> the generic type
 * @param <B> the generic type
 * @param <O> the generic type
 */
public abstract class OptionOptimizer<C extends Classifier<C>, B extends ClassifierBuilder<C, B>, O extends Option> {

	/** The scorer. */
	protected final ObjectiveScorer<BinaryClassificationTarget> scorer;
	
	/** The builder. */
	protected final B builder;
	
	/** The instances. */
	protected final BinaryClassificationInstances instances;

	/** The Constant DEFAULT_FOLDS. */
	protected static final int DEFAULT_FOLDS = 5;
	
	/** The folds. */
	protected final int folds;

	/** The best objective. */
	protected double bestObjective;

	/**
	 * Instantiates a new option optimizer.
	 *
	 * @param scorer the scorer
	 * @param builder the builder
	 * @param instances the instances
	 * @param folds the folds
	 */
	protected OptionOptimizer(
			ObjectiveScorer<BinaryClassificationTarget> scorer, B builder,
			BinaryClassificationInstances instances, int folds) {
		this.scorer = scorer;
		this.builder = builder;
		this.instances = instances;
		checkArgument(folds > 1, "folds must be > 1, given: %s", folds);
		this.folds = folds;
	}

	/**
	 * Instantiates a new option optimizer.
	 *
	 * @param scorer the scorer
	 * @param builder the builder
	 * @param instances the instances
	 */
	protected OptionOptimizer(
			ObjectiveScorer<BinaryClassificationTarget> scorer, B builder,
			BinaryClassificationInstances instances) {
		this(scorer, builder, instances, DEFAULT_FOLDS);
	}

	/**
	 * resets iterators for subclass; used for sequential iteration.
	 */
	public abstract void reset();

	/**
	 * The Class AbstractBooleanOptionOptimizer.
	 *
	 * @param <C> the generic type
	 * @param <B> the generic type
	 */
	protected static abstract class AbstractBooleanOptionOptimizer<C extends Classifier<C>, B extends ClassifierBuilder<C, B>>
			extends OptionOptimizer<C, B, BooleanOption> {

		/** The opt. */
		protected final BooleanOption opt;

		/**
		 * Instantiates a new abstract boolean option optimizer.
		 *
		 * @param scorer the scorer
		 * @param builder the builder
		 * @param instances the instances
		 * @param opt the opt
		 * @param folds the folds
		 */
		protected AbstractBooleanOptionOptimizer(
				ObjectiveScorer<BinaryClassificationTarget> scorer, B builder,
				BinaryClassificationInstances instances, BooleanOption opt,
				int folds) {
			super(scorer, builder, instances, folds);
			this.opt = opt;
		}

		/**
		 * Instantiates a new abstract boolean option optimizer.
		 *
		 * @param scorer the scorer
		 * @param builder the builder
		 * @param instances the instances
		 * @param opt the opt
		 */
		protected AbstractBooleanOptionOptimizer(
				ObjectiveScorer<BinaryClassificationTarget> scorer, B builder,
				BinaryClassificationInstances instances, BooleanOption opt) {
			super(scorer, builder, instances);
			this.opt = opt;
		}

		/**
		 * Optimize.
		 *
		 * @return the boolean
		 */
		public abstract Boolean optimize();
	}

	/**
	 * The Class AbstractFloatOptionGridOptimizer.
	 *
	 * @param <C> the generic type
	 * @param <B> the generic type
	 */
	protected static abstract class AbstractFloatOptionGridOptimizer<C extends Classifier<C>, B extends ClassifierBuilder<C, B>>
			extends OptionOptimizer<C, B, FloatOption> {

		/** The opt. */
		protected final FloatOption opt;

		/**
		 * Instantiates a new abstract float option grid optimizer.
		 *
		 * @param scorer the scorer
		 * @param builder the builder
		 * @param instances the instances
		 * @param opt the opt
		 * @param folds the folds
		 */
		protected AbstractFloatOptionGridOptimizer(
				ObjectiveScorer<BinaryClassificationTarget> scorer, B builder,
				BinaryClassificationInstances instances, FloatOption opt,
				int folds) {
			super(scorer, builder, instances, folds);
			this.opt = opt;
		}

		/**
		 * Instantiates a new abstract float option grid optimizer.
		 *
		 * @param scorer the scorer
		 * @param builder the builder
		 * @param instances the instances
		 * @param opt the opt
		 */
		protected AbstractFloatOptionGridOptimizer(
				ObjectiveScorer<BinaryClassificationTarget> scorer, B builder,
				BinaryClassificationInstances instances, FloatOption opt) {
			super(scorer, builder, instances);
			this.opt = opt;
		}

		/**
		 * Optimize.
		 *
		 * @return the double
		 */
		public abstract Double optimize();
	}

	/**
	 * The Class AbstractIntegerOptionGridOptimizer.
	 *
	 * @param <C> the generic type
	 * @param <B> the generic type
	 */
	protected static abstract class AbstractIntegerOptionGridOptimizer<C extends Classifier<C>, B extends ClassifierBuilder<C, B>>
			extends OptionOptimizer<C, B, FloatOption> {

		/** The opt. */
		protected final IntegerOption opt;

		/**
		 * Instantiates a new abstract integer option grid optimizer.
		 *
		 * @param scorer the scorer
		 * @param builder the builder
		 * @param instances the instances
		 * @param opt the opt
		 * @param folds the folds
		 */
		protected AbstractIntegerOptionGridOptimizer(
				ObjectiveScorer<BinaryClassificationTarget> scorer, B builder,
				BinaryClassificationInstances instances, IntegerOption opt,
				int folds) {
			super(scorer, builder, instances, folds);
			this.opt = opt;
		}

		/**
		 * Instantiates a new abstract integer option grid optimizer.
		 *
		 * @param scorer the scorer
		 * @param builder the builder
		 * @param instances the instances
		 * @param opt the opt
		 */
		protected AbstractIntegerOptionGridOptimizer(
				ObjectiveScorer<BinaryClassificationTarget> scorer, B builder,
				BinaryClassificationInstances instances, IntegerOption opt) {
			super(scorer, builder, instances);
			this.opt = opt;
		}

		/**
		 * Optimize.
		 *
		 * @return the integer
		 */
		public abstract Integer optimize();
	}

	/**
	 * The Class AbstractEnumOptionGridOptimizer.
	 *
	 * @param <C> the generic type
	 * @param <B> the generic type
	 * @param <T> the generic type
	 */
	protected static abstract class AbstractEnumOptionGridOptimizer<C extends Classifier<C>, B extends ClassifierBuilder<C, B>, T extends Enum<T>>
			extends OptionOptimizer<C, B, FloatOption> {

		/** The opt. */
		protected final EnumOption<T> opt;

		/**
		 * Instantiates a new abstract enum option grid optimizer.
		 *
		 * @param scorer the scorer
		 * @param builder the builder
		 * @param instances the instances
		 * @param opt the opt
		 * @param folds the folds
		 */
		protected AbstractEnumOptionGridOptimizer(
				ObjectiveScorer<BinaryClassificationTarget> scorer, B builder,
				BinaryClassificationInstances instances, EnumOption<T> opt,
				int folds) {
			super(scorer, builder, instances, folds);
			this.opt = opt;
		}

		/**
		 * Instantiates a new abstract enum option grid optimizer.
		 *
		 * @param scorer the scorer
		 * @param builder the builder
		 * @param instances the instances
		 * @param opt the opt
		 */
		protected AbstractEnumOptionGridOptimizer(
				ObjectiveScorer<BinaryClassificationTarget> scorer, B builder,
				BinaryClassificationInstances instances, EnumOption<T> opt) {
			super(scorer, builder, instances);
			this.opt = opt;
		}

		/**
		 * Optimize.
		 *
		 * @return the t
		 */
		public abstract T optimize();
	}

	/**
	 * The Class AbstractConfigurableOptionGridOptimizer.
	 *
	 * @param <C> the generic type
	 * @param <B> the generic type
	 * @param <T> the generic type
	 */
	protected static abstract class AbstractConfigurableOptionGridOptimizer<C extends Classifier<C>, B extends ClassifierBuilder<C, B>, T extends Configurable<T>>
			extends OptionOptimizer<C, B, FloatOption> {

		/** The opt. */
		protected final ConfigurableOption<T> opt;

		/**
		 * Instantiates a new abstract configurable option grid optimizer.
		 *
		 * @param scorer the scorer
		 * @param builder the builder
		 * @param instances the instances
		 * @param opt the opt
		 * @param folds the folds
		 */
		protected AbstractConfigurableOptionGridOptimizer(
				ObjectiveScorer<BinaryClassificationTarget> scorer, B builder,
				BinaryClassificationInstances instances,
				ConfigurableOption<T> opt, int folds) {
			super(scorer, builder, instances, folds);
			this.opt = opt;
		}

		/**
		 * Instantiates a new abstract configurable option grid optimizer.
		 *
		 * @param scorer the scorer
		 * @param builder the builder
		 * @param instances the instances
		 * @param opt the opt
		 */
		protected AbstractConfigurableOptionGridOptimizer(
				ObjectiveScorer<BinaryClassificationTarget> scorer, B builder,
				BinaryClassificationInstances instances,
				ConfigurableOption<T> opt) {
			super(scorer, builder, instances);
			this.opt = opt;
		}

		/**
		 * Optimize.
		 *
		 * @return the configuration
		 */
		public abstract Configuration<T> optimize();
	}

}
