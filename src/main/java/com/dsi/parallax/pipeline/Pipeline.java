/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

import java.io.Serializable;
import java.util.Iterator;
import java.util.List;

import com.google.common.collect.Lists;

/**
 * Class to handle serial data processing. Its a container class which connects 
 * a sequence of pipes together, which serially processes data from, say, a raw file 
 * format to something that is appropriate for binary classification 
 * @author jattenberg 
 */
public class Pipeline<I, O> implements Serializable {
    private static final long serialVersionUID = 4827997256416927082L;
    private Source<I> source;
    private List<Pipe<?, ?>> pipeline;

    private Pipeline() {
        pipeline = Lists.newLinkedList();
    }

    private Pipeline(List<Pipe<?, ?>> pipeline) {
        this.pipeline = Lists.newLinkedList(pipeline);
    }

    /**
     * enforce that the first intermediate type is the output of the source
     * 
     * @return new pipeline
     */
    public static <I, O> Pipeline<I, I> newPipeline() {
        return new Pipeline<I, I>();
    }

    /**
     * constructor that enforced the first intermediate type is output of the
     * source
     * 
     * @param source
     * @return new pipe
     */
    public static <I> Pipeline<I, I> newPipeline(Source<I> source) {
        Pipeline<I, I> p = new Pipeline<I, I>();
        p.setSource(source);
        return p;
    }

    /**
     * method chaining setter for adding a source
     * 
     * @param source
     * @return the pipeline
     */
    public Pipeline<I, O> setSource(Source<I> source) {
        this.source = source;
        return this;
    }

    /**
     * typesafe addition of new pipes to pipeline
     * 
     * @param pipe
     * @return pipeline with the new pipe
     */
    public <T> Pipeline<I, T> addPipe(Pipe<O, T> pipe) {
    	// create a new pipeline object with source type <I> and new target type <T>
    	// and add the existing pipeline to it
        Pipeline<I, T> out = new Pipeline<I, T>(this.pipeline);
        // add the new pipe to the new pipeline
        out.pipeline.add(pipe);
        out.source = source;
        return out;
    }

    /**
     * The method to process the current pipeline. It steps through the pipes in the 
     * pipeline. At each step it passes the source type context iterator through the pipe 
     * to generate a target type context iterator. 
     * @return an iterator of the context of the final output type 
     */
    @SuppressWarnings("unchecked")
    public Iterator<Context<O>> process() {
        @SuppressWarnings("rawtypes")
        Iterator it = source.provideData();
        for (@SuppressWarnings("rawtypes")
        Pipe pipe : pipeline) {
            it = pipe.processIterator(it);
        }
        return it;
    }

	public Source<I> getSource() {
		return source;
	}

	public List<Pipe<?, ?>> getPipeline() {
		return pipeline;
	}
    
}
