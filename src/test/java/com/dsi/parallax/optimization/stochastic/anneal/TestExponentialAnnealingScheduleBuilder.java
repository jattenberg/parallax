package com.dsi.parallax.optimization.stochastic.anneal;

import static org.junit.Assert.*;

import org.junit.Test;

import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.optimization.stochastic.anneal.AnnealingScheduleBuilder;
import com.dsi.parallax.optimization.stochastic.anneal.AnnealingScheduleType;
import com.dsi.parallax.optimization.stochastic.anneal.ExponentialAnnealingSchedule;
import com.dsi.parallax.optimization.stochastic.anneal.ExponentialAnnealingScheduleBuilder;

public class TestExponentialAnnealingScheduleBuilder {

	@Test
	public void testCorrectType() {
		ExponentialAnnealingScheduleBuilder builder = new ExponentialAnnealingScheduleBuilder();

		assertEquals(builder.correspondingType(),
				AnnealingScheduleType.EXPONENTIAL);
	}

	@Test
	public void testConfiguresInitialRate() {
		ExponentialAnnealingScheduleBuilder builder = new ExponentialAnnealingScheduleBuilder();

		builder.setInitialRate(5.0);
		assertEquals(builder.getInitialLearningRate(), 5, 0.);
		ExponentialAnnealingSchedule schedule = builder.build();
		assertEquals(schedule.getInitialLearningRate(), 5, 0.);

		Configuration<AnnealingScheduleBuilder> configuration = new Configuration<AnnealingScheduleBuilder>(
				ExponentialAnnealingScheduleBuilder.options);
		configuration.addFloatValueOnShortName("r", 100);

		builder = new ExponentialAnnealingScheduleBuilder(configuration);
		assertEquals(builder.getInitialLearningRate(), 100, 0.);
		schedule = builder.build();
		assertEquals(schedule.getInitialLearningRate(), 100, 0.);

		configuration.addFloatValueOnShortName("r", 155);
		builder.configure(configuration);

		assertEquals(builder.getInitialLearningRate(), 155, 0.);
		schedule = builder.build();
		assertEquals(schedule.getInitialLearningRate(), 155, 0.);
	}

	@Test
	public void testConfiguresExponentialBase() {
		ExponentialAnnealingScheduleBuilder builder = new ExponentialAnnealingScheduleBuilder();

		builder.setExponentialBase(.5);
		assertEquals(builder.getExponentialBase(), .5, 0.);
		ExponentialAnnealingSchedule schedule = builder.build();
		assertEquals(schedule.getExponentialBase(), .5, 0.);

		Configuration<AnnealingScheduleBuilder> configuration = new Configuration<AnnealingScheduleBuilder>(
				ExponentialAnnealingScheduleBuilder.options);
		configuration.addFloatValueOnShortName("b", .100);

		builder = new ExponentialAnnealingScheduleBuilder(configuration);
		assertEquals(builder.getExponentialBase(), .100, 0.);
		schedule = builder.build();
		assertEquals(schedule.getExponentialBase(), .100, 0.);

		configuration.addFloatValueOnShortName("b", .155);
		builder.configure(configuration);

		assertEquals(builder.getExponentialBase(), .155, 0.);
		schedule = builder.build();
		assertEquals(schedule.getExponentialBase(), .155, 0.);
	}
}
