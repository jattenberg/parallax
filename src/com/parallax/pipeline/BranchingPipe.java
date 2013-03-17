/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline;


/**
 * pipe that
 * passes data through several pipes in parrallel
 * the results of this processing is combined using a combiner class
 * each of the component pipes must have the same in and out format,
 * enforced by I and C
 * the combiner takes these C's and turns them into a O
 * 
 * note that all branches need to produce the same number outputs for 
 * each input!!!
 * 
 * currently applying labels in branches is tricky..
 * 
 * several pipes can be put into a branch pipe using MultiPipe
 * 
 * @author jattenberg
 *
 * @param <I>
 * @param <C>
 * @param <O>
 */
public interface BranchingPipe<I,C,O> extends Pipe<I,O> {

    public void addBranch(Pipe<I,C> pipe);
    public boolean removeBranch(Pipe<I,C> pipe);
    public int size();
    public void addCombiner(Combiner<C,O> combiner);
}
