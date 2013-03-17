package com.parallax.optimization.stochastic.anneal;

import static org.junit.Assert.*;

import org.junit.Test;

import com.parallax.ml.util.option.Configuration;

public class TestAdaGradAnnealingScheduleBuilder {

	@Test
	public void testCorrectType() {
		AdaGradAnnealingScheduleBuilder builder = new AdaGradAnnealingScheduleBuilder();

		assertEquals(builder.correspondingType(), AnnealingScheduleType.ADAGRAD);
	}

	@Test
	public void testConfiguresInitialRate() {
		AdaGradAnnealingScheduleBuilder builder = new AdaGradAnnealingScheduleBuilder();

		builder.setInitialRate(5.0);
		assertEquals(builder.getInitialLearningRate(), 5, 0.);
		AdaGradAnnealingSchedule schedule = builder.build();
		assertEquals(schedule.getInitialLearningRate(), 5, 0.);

		Configuration<AnnealingScheduleBuilder> configuration = new Configuration<AnnealingScheduleBuilder>(
				AdaGradAnnealingScheduleBuilder.options);
		configuration.addFloatValueOnShortName("r", 100);

		builder = new AdaGradAnnealingScheduleBuilder(configuration);
		assertEquals(builder.getInitialLearningRate(), 100, 0.);
		schedule = builder.build();
		assertEquals(schedule.getInitialLearningRate(), 100, 0.);

		configuration.addFloatValueOnShortName("r", 155);
		builder.configure(configuration);

		assertEquals(builder.getInitialLearningRate(), 155, 0.);
		schedule = builder.build();
		assertEquals(schedule.getInitialLearningRate(), 155, 0.);
	}

}
