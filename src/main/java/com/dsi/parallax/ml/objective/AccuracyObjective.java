/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.objective;

import com.dsi.parallax.ml.evaluation.ConfusionMatrix;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.model.Model;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;

/**
 * estimates the performance of a model on the supplied instances
 * using accuracy; the percentage of examples that have been correctly
 * labeled by the model. 
 * 
 * Note that accuracy estimates used in this objective are "soft" estimates,
 * recognizing the percent of the probability mass given to the correct class.
 * 
 *
 * @author jattenberg
 */
public class AccuracyObjective implements Objective<BinaryClassificationTarget>{

	/* (non-Javadoc)
	 * @see com.parallax.ml.objective.Objective#evaluate(java.lang.Iterable, com.parallax.ml.model.Model)
	 */
	@Override
	public <I extends Instance<BinaryClassificationTarget>, E extends Iterable<I>, M extends Model<BinaryClassificationTarget, M>> double evaluate(
			E instances, M model) {
		ConfusionMatrix conf = new ConfusionMatrix(2);
		
		for(I inst : instances)
			conf.addInfo(inst.getLabel(), model.predict(inst));
		return conf.computeAccuracy();
	}

}
