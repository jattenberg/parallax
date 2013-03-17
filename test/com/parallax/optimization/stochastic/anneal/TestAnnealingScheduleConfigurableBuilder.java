package com.parallax.optimization.stochastic.anneal;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.google.common.collect.Sets;
import com.parallax.ml.util.option.ParentNestedConfigurableTestUtils;

public class TestAnnealingScheduleConfigurableBuilder {

	@Test
	public void parentTests() {
		AnnealingScheduleConfigurableBuilder configbuilder = new AnnealingScheduleConfigurableBuilder();

		ParentNestedConfigurableTestUtils.testMatchesConfig(
				AnnealingScheduleType.class, configbuilder);
		ParentNestedConfigurableTestUtils.testBuildsCorrectType(
				AnnealingScheduleType.class, configbuilder);
		ParentNestedConfigurableTestUtils.testNestedConfigurations(
				AnnealingScheduleType.class, configbuilder);
	}

	@Test
	public void testBuilds() {
		testBuilds(AnnealingScheduleType.CONSTANT);
		testBuilds(AnnealingScheduleType.EXPONENTIAL);
		testBuilds(AnnealingScheduleType.INVERSE);
		testBuilds(AnnealingScheduleType.ADAGRAD);
	}

	public static void testBuilds(AnnealingScheduleType type) {
		AnnealingScheduleConfigurableBuilder configbuilder = new AnnealingScheduleConfigurableBuilder();
		assertTrue(Sets.newHashSet(configbuilder.childTypes()).contains(type));

		configbuilder.setConfigurableType(type);

		AnnealingSchedule schedlue = configbuilder.build();
		AnnealingScheduleBuilder builder = configbuilder
				.currentNestedConfigurable();

		assertEquals(type.getConfigurable().getClass().getName(), builder
				.getClass().getName());
		assertEquals(schedlue.getClass().getName(), builder.build().getClass()
				.getName());
	}
}
