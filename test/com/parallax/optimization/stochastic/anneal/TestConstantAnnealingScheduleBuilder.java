package com.parallax.optimization.stochastic.anneal;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.parallax.ml.util.option.Configuration;

public class TestConstantAnnealingScheduleBuilder {

	@Test
	public void testCorrectType() {
		ConstantAnnealingScheduleBuilder builder = new ConstantAnnealingScheduleBuilder();
		assertEquals(builder.correspondingType(),
				AnnealingScheduleType.CONSTANT);
	}

	@Test
	public void testConfiguresInitialRate() {

		ConstantAnnealingScheduleBuilder builder = new ConstantAnnealingScheduleBuilder();

		builder.setInitialRate(5);
		assertEquals(builder.getLearningRate(), 5, 0.);
		ConstantAnnealingSchedule schedule = builder.build();
		assertEquals(schedule.getInitialLearningRate(), 5, 0.);

		Configuration<AnnealingScheduleBuilder> configuration = new Configuration<AnnealingScheduleBuilder>(
				ConstantAnnealingScheduleBuilder.options);
		configuration.addFloatValueOnShortName("r", 100);

		builder = new ConstantAnnealingScheduleBuilder(configuration);
		assertEquals(builder.getLearningRate(), 100, 0.);
		schedule = builder.build();
		assertEquals(schedule.getInitialLearningRate(), 100, 0.);

		configuration.addFloatValueOnShortName("r", 155);
		builder.configure(configuration);

		assertEquals(builder.getLearningRate(), 155, 0.);
		schedule = builder.build();
		assertEquals(schedule.getInitialLearningRate(), 155, 0.);
	}
}
