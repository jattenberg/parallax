package com.parallax.ml.classifier.smoother;

import com.parallax.ml.util.SigmoidType;
import com.parallax.ml.util.pair.PrimitivePair;

/**
 * The Class LogitSmoother, very simply runs the incoming prediction value
 * through a logit function. A baseline smoother similar to NullSmoother.
 */
public class LogitSmoother extends
		AbstractUpdateableSmoother<LogitSmoother> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 8890541405836911872L;

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.smoother.UpdateableSmoother#update(com
	 * .parallax.ml.util.pair.PrimitivePair)
	 */
	@Override
	public void update(PrimitivePair p) {
		// nothing to do!
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.smoother.Smoother#smooth(double)
	 */
	@Override
	public double smooth(double prediction) {
		return SigmoidType.LOGIT.sigmoid(prediction);
	}

}
