package com.dsi.parallax.ml.classifier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import java.util.Set;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.Classifiers;
import com.dsi.parallax.ml.classifier.ClassifierBuilder.ClassifierOptions;
import com.google.common.collect.Sets;

/**
 * The Class TestClassifiers.
 */
public class TestClassifiers {

	/**
	 * Test enums match.
	 */
	@Test
	public void testEnumsMatch() {
		for (Classifiers classifiers : Classifiers.values()) {
			if (classifiers.equals(Classifiers.ID3))
				continue;
			ClassifierOptions<?, ?> conf = (ClassifierOptions<?, ?>) classifiers
					.getOptions();
			assertEquals(classifiers, conf.getClassifierType());
		}
	}

	/**
	 * Test strings are different.
	 */
	@Test
	public void testStringsAreDifferent() {
		Set<String> stringSet = Sets.newHashSet();
		for (Classifiers classifiers : Classifiers.values()) {
			assertFalse(stringSet.contains(classifiers.getName()));
			stringSet.add(classifiers.getName());
		}

	}

	/**
	 * Test classes are different.
	 */
	@Test
	public void testClassesAreDifferent() {
		Set<Class<?>> stringSet = Sets.newHashSet();
		for (Classifiers classifiers : Classifiers.values()) {
			assertFalse(stringSet.contains(classifiers.getClass()));
			stringSet.add(classifiers.getClass());
		}
		assertEquals(Classifiers.values().length, stringSet.size());
	}

	/**
	 * Test builders are different.
	 */
	@Test
	public void testBuildersAreDifferent() {
		Set<Class<?>> stringSet = Sets.newHashSet();
		for (Classifiers classifiers : Classifiers.values()) {
			assertFalse(stringSet.contains(classifiers.getClassifierBuilder(1,
					true).getClass()));
			stringSet.add(classifiers.getClassifierBuilder(1, true).getClass());
		}
		assertEquals(Classifiers.values().length, stringSet.size());
	}

	/**
	 * Test classifiers are different.
	 */
	@Test
	public void testClassifiersAreDifferent() {
		Set<Class<?>> stringSet = Sets.newHashSet();
		for (Classifiers classifiers : Classifiers.values()) {
			assertFalse(stringSet.contains(classifiers.getClassifier(1, true)
					.getClass()));
			stringSet.add(classifiers.getClassifier(1, true).getClass());
		}
		assertEquals(Classifiers.values().length, stringSet.size());
	}

}
