package com.dsi.parallax.optimization.stochastic.anneal;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.dsi.parallax.ml.util.option.ParentNestedConfigurableTestUtils;
import com.dsi.parallax.optimization.stochastic.anneal.AnnealingSchedule;
import com.dsi.parallax.optimization.stochastic.anneal.AnnealingScheduleBuilder;
import com.dsi.parallax.optimization.stochastic.anneal.AnnealingScheduleConfigurableBuilder;
import com.dsi.parallax.optimization.stochastic.anneal.AnnealingScheduleType;
import com.google.common.collect.Sets;

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
