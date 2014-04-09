/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.discretization;

import java.util.Map;
import java.util.Set;

import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.pipeline.AbstractAccumulatingPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;

/**
 * base class for discretizing continuous variables; takes a continuous variable
 * and replaces it with a categorical variable. Functionally, this takes a var, 
 * and replaces with by a number of binary features indicated by the expansionFactor
 * construction parameter. One of those binary features is set indicating the bucket 
 * a particular instance of a continuous feature belongs to. 
 * 
 *
 * @author jattenberg
 */
public abstract class AbstractDiscretizationPipe extends
		AbstractAccumulatingPipe<LinearVector, LinearVector> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 5794789649211282487L;
	
	/** what are the discrete features we are going to consider?. */
	protected transient Set<Integer> examinedFeatures;
	
	/** we turn each continuous feature into this many discrete features. */
	protected final int expansionFactor;
	
	/** The index to bounds. */
	protected final Map<Integer, DiscreteBounds> indexToBounds;
	
	/** The keep continuous. */
	private final boolean keepContinuous;

	/**
	 * Instantiates a new abstract discretization pipe.
	 *
	 * @param expansionFactor the expansion factor; number of discrete features for each continuous feature
	 * @param readForInitialization the number of examples to consider for building the discretization data structures. 
	 * @param keepContinuous the should we descretize AND keep continuous features? 
	 */
	protected AbstractDiscretizationPipe(int expansionFactor, int readForInitialization,
			boolean keepContinuous) {
		super(readForInitialization);
		this.expansionFactor = expansionFactor;
		this.keepContinuous = keepContinuous;
		indexToBounds = Maps.newHashMap();
	}

	/**
	 * Instantiates a new abstract discretization pipe.
	 * considers ALL available examples for initialization
	 *
	 * @param expansionFactor the expansion factor; number of discrete features for each continuous feature
	 * @param readForInitialization the number of examples to consider for building the discretization data structures. 
	 */
	protected AbstractDiscretizationPipe(int expansionFactor,
			boolean keepContinuous) {
		this(expansionFactor, -1, keepContinuous);
	}


	/**
	 * Instantiates a new abstract discretization pipe.
	 *
	 * @param expansionFactor the expansion factor; number of discrete features for each continuous feature
	 * @param keepContinuous the should we descretize AND keep continuous features? 
	 * @param examinedFeatures what features should be discretized? 
	 */
	protected AbstractDiscretizationPipe(int expansionFactor,
			boolean keepContinuous, Set<Integer> examinedFeatures) {
		this(expansionFactor, keepContinuous);
		this.examinedFeatures = Sets.newTreeSet(examinedFeatures);
	}

	/**
	 * Instantiates a new abstract discretization pipe.
	 *
	 * @param examinedFeatures what features should be discretized? 
	 * @param expansionFactor the expansion factor; number of discrete features for each continuous feature
	 * @param readForInitialization the number of examples to consider for building the discretization data structures. 
	 * @param keepContinuous the should we descretize AND keep continuous features? 
	 */
	protected AbstractDiscretizationPipe(Set<Integer> examinedFeatures,
			int expansionFactor, boolean keepContinuous, int readToInitialize) {
		this(expansionFactor, readToInitialize, keepContinuous);
		this.examinedFeatures = Sets.newTreeSet(examinedFeatures);
	}

	/* (non-Javadoc)
	 * @see com.parallax.pipeline.AbstractAccumulatingPipe#operate(com.parallax.pipeline.Context)
	 */
	protected Context<LinearVector> operate(Context<LinearVector> context) {
		LinearVector in = context.getData();

		LinearVector out = LinearVectorFactory.getVector(in.size()
				+ expansionFactor * indexToBounds.size());
		for (int x_i : in) {
			if (!indexToBounds.containsKey(x_i) || keepContinuous)
				out.resetValue(x_i, in.getValue(x_i));
			if (indexToBounds.containsKey(x_i)) {
				int startPoint = indexToBounds.get(x_i).getStartingPoint();
				int bucket = indexToBounds.get(x_i)
						.discretize(in.getValue(x_i));
				out.resetValue(startPoint + bucket, 1);
			}
		}

		return Context.createContext(context, out);
	}

}
