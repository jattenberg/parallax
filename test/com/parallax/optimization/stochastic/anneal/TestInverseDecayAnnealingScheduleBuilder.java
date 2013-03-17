package com.parallax.optimization.stochastic.anneal;

import static org.junit.Assert.*;

import org.junit.Test;

import com.parallax.ml.util.option.Configuration;

public class TestInverseDecayAnnealingScheduleBuilder {

	@Test
	public void testCorrectType() {
		InverseDecayAnnealingScheduleBuilder builder = new InverseDecayAnnealingScheduleBuilder();

		assertEquals(builder.correspondingType(), AnnealingScheduleType.INVERSE);
	}

	@Test
	public void testConfiguresInitialRate() {
		InverseDecayAnnealingScheduleBuilder builder = new InverseDecayAnnealingScheduleBuilder();

		builder.setInitialRate(5.0);
		assertEquals(builder.getInitialLearningRate(), 5, 0.);
		InverseDecayAnnealingSchedule schedule = builder.build();
		assertEquals(schedule.getInitialLearningRate(), 5, 0.);

		Configuration<AnnealingScheduleBuilder> configuration = new Configuration<AnnealingScheduleBuilder>(
				InverseDecayAnnealingScheduleBuilder.options);
		configuration.addFloatValueOnShortName("r", 100);

		builder = new InverseDecayAnnealingScheduleBuilder(configuration);
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
	public void testConfiguresDecayRate() {
		InverseDecayAnnealingScheduleBuilder builder = new InverseDecayAnnealingScheduleBuilder();

		builder.setDecay(5.0);
		assertEquals(builder.getDecay(), 5, 0.);
		InverseDecayAnnealingSchedule schedule = builder.build();
		assertEquals(schedule.getDecay(), 5, 0.);

		Configuration<AnnealingScheduleBuilder> configuration = new Configuration<AnnealingScheduleBuilder>(
				InverseDecayAnnealingScheduleBuilder.options);
		configuration.addFloatValueOnShortName("d", 100);

		builder = new InverseDecayAnnealingScheduleBuilder(configuration);
		assertEquals(builder.getDecay(), 100, 0.);
		schedule = builder.build();
		assertEquals(schedule.getDecay(), 100, 0.);

		configuration.addFloatValueOnShortName("d", 155);
		builder.configure(configuration);

		assertEquals(builder.getDecay(), 155, 0.);
		schedule = builder.build();
		assertEquals(schedule.getDecay(), 155, 0.);
	}
}
