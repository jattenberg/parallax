package com.dsi.parallax.optimization.stochastic.anneal;

import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.NestedConfigurableType;

public enum AnnealingScheduleType implements
		NestedConfigurableType<AnnealingScheduleBuilder, AnnealingScheduleType> {

	CONSTANT {

		@Override
		public AnnealingScheduleBuilder getConfigurable() {
			return new ConstantAnnealingScheduleBuilder();
		}

		@Override
		public Configuration<AnnealingScheduleBuilder> getDefaultConfiguration() {
			return new Configuration<AnnealingScheduleBuilder>(
					ConstantAnnealingScheduleBuilder.options);
		}
	},
	EXPONENTIAL {

		@Override
		public AnnealingScheduleBuilder getConfigurable() {
			return new ExponentialAnnealingScheduleBuilder();
		}

		@Override
		public Configuration<AnnealingScheduleBuilder> getDefaultConfiguration() {
			return new Configuration<AnnealingScheduleBuilder>(
					ExponentialAnnealingScheduleBuilder.options);
		}
	},
	INVERSE {

		@Override
		public AnnealingScheduleBuilder getConfigurable() {
			return new InverseDecayAnnealingScheduleBuilder();
		}

		@Override
		public Configuration<AnnealingScheduleBuilder> getDefaultConfiguration() {
			return new Configuration<AnnealingScheduleBuilder>(
					InverseDecayAnnealingScheduleBuilder.options);
		}

	},
	ADAGRAD {

		@Override
		public AnnealingScheduleBuilder getConfigurable() {
			return new AdaGradAnnealingScheduleBuilder();
		}

		@Override
		public Configuration<AnnealingScheduleBuilder> getDefaultConfiguration() {
			return new Configuration<AnnealingScheduleBuilder>(
					AdaGradAnnealingScheduleBuilder.options);
		}

	}
}
