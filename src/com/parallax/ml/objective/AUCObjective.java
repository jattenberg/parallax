/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.objective;

import com.parallax.ml.evaluation.ReceiverOperatingCharacteristic;
import com.parallax.ml.instance.Instance;
import com.parallax.ml.model.Model;
import com.parallax.ml.target.BinaryClassificationTarget;

/**
 * estimates the performance of a model on the supplied instances
 * using AUC, the area under the ROC curve.
 * 
 * @see {@link ReceiverOperatingCharacteristic}
 *
 * @author jattenberg
 */
public class AUCObjective implements Objective<BinaryClassificationTarget>{

	/* (non-Javadoc)
	 * @see com.parallax.ml.objective.Objective#evaluate(java.lang.Iterable, com.parallax.ml.model.Model)
	 */
	@Override
	public <I extends Instance<BinaryClassificationTarget>, E extends Iterable<I>, M extends Model<BinaryClassificationTarget, M>> double evaluate(
			E instances, M model) {
		ReceiverOperatingCharacteristic ROC = new ReceiverOperatingCharacteristic();
		
		for(I inst : instances)
			ROC.add(inst.getLabel(), model.predict(inst));
		return ROC.binaryAUC();
	}

}
