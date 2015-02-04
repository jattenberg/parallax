package com.dsi.parallax.ml.classifier.lazy;

import com.dsi.parallax.ml.classifier.linear.updateable.LogisticRegression;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.KDTree.Entry;

import java.util.List;

public class LocalLogisticRegression extends
		AbstractUpdateableKDTreeClassifier<LocalLogisticRegression> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -9141885430862201440L;

	public LocalLogisticRegression(int dimension, boolean bias) {
		super(dimension, bias);
		initialize();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.AbstractClassifier#regress(com.parallax.ml
	 * .instance.Instanze)
	 */
	@Override
	protected double regress(Instance<?> inst) {
		List<Entry<BinaryClassificationTarget>> neighborLabels = findNeighbors(inst);
		if (neighborLabels.size() == 0)
			return 0.5;
		BinaryClassificationInstances insts = buildInstances(neighborLabels);
		LogisticRegression LR = new LogisticRegression(dimension, true);
		LR.train(insts);
		return LR.predict(inst).getValue();
	}

	private BinaryClassificationInstances buildInstances(
			List<Entry<BinaryClassificationTarget>> neighborLabels) {
		BinaryClassificationInstances insts = new BinaryClassificationInstances(
				dimension);
		for (Entry<BinaryClassificationTarget> inst : neighborLabels) {
			insts.addInstance(new BinaryClassificationInstance(inst.value,
					inst.position));
		}

		return insts;
	}

	@Override
	protected LocalLogisticRegression getModel() {
		return this;
	}
}
