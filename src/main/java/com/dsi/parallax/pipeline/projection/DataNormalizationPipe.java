package com.dsi.parallax.pipeline.projection;


import com.dsi.parallax.ml.projection.DataNormalization;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.pipeline.AbstractAccumulatingPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.common.collect.Collections2;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.List;

// TODO: Auto-generated Javadoc
/**
 * A pipe that normalizes the linear vectors by subtracting each feature 
 * value by its mean and dividing the result by standard deviation. The 
 * end result is that all features have zero mean and unit standard 
 * deviation
 *
 * @author spchopra
 */
public class DataNormalizationPipe extends AbstractAccumulatingPipe<LinearVector, LinearVector> {

    /** The Constant serialVersionUID. */
	private static final long serialVersionUID = 6805806706173300839L;

	/** The DataNormalization object */
    private DataNormalization dnorm;
    
    /**
     * Instantiates a new DataNormalizationPipe
     *
     * @param dn the DataNormalization object
     */
    public DataNormalizationPipe(DataNormalization dn) {
        this(dn, -1);
    }

    /**
     * Instantiates a new DataNormalizationPipe
     *
     * @param dn	the DataNormalization object
     * @param toConsider flag indicating whether the DataNormalization
     * 	  object has pre-computed mean and standard deviation or not
     * 					
     */
    public DataNormalizationPipe(DataNormalization dn, int toConsider) {
        super(toConsider);
        this.dnorm = dn;
    }
    
    /**
     * Returns a flag indicating whether the DataNormalization object 
     * has been trained or not
     */
    @Override
    public boolean isTrained() {
        return dnorm.isBuilt();
    }
    
	/* (non-Javadoc)
	 * @see com.parallax.pipeline.Pipe#getType()
	 */
	@Override
	public Type getType() {
		return new TypeToken<DataNormalizationPipe>(){}.getType();
	}

	/* (non-Javadoc)
	 * @see com.parallax.pipeline.AbstractAccumulatingPipe#operate(com.parallax.pipeline.Context)
	 */
	@Override
	protected Context<LinearVector> operate(Context<LinearVector> context) {
		return Context.createContext(context, dnorm.project(context.getData()));
	}

	/* (non-Javadoc)
	 * @see com.parallax.pipeline.AbstractAccumulatingPipe#batchProcess(java.util.List)
	 */
	@Override
	protected void batchProcess(List<Context<LinearVector>> infoList) {
		dnorm.build(Collections2.transform(infoList, uncontextifier));
	}
}
