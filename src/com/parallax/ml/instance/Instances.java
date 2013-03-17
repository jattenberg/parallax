/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.instance;

import static com.google.common.base.Preconditions.checkArgument;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import com.google.common.collect.Lists;
import com.parallax.ml.projection.Projection;
import com.parallax.ml.util.pair.GenericPair;
import com.parallax.ml.vector.LinearVector;

// TODO: Auto-generated Javadoc
/**
 * The abstract Instances class which handles a collection of {@link Instance} objects. 
 * These objects are stored in a List
 * 
 * @param <I>
 *            the generic type
 */
public abstract class Instances<I extends Instance<?>> implements Serializable,
		Iterable<I>, Collection<I> {

	/** The Constant serialVersionUID. */
	static final long serialVersionUID = 2L;

	/** The instances. */
	protected List<I> instances;

	/** The dimensions. */
	protected final int dimensions;

	/**
	 * Instantiates a new instances.
	 * 
	 * @param dimensions
	 *            the dimensions
	 * @param input
	 *            the input
	 */
	protected Instances(int dimensions, I... input) {
		this(dimensions);
		this.instances = new ArrayList<I>();
		for (I in : input)
			addInstance(in);
	}

	/**
	 * Instantiates a new instances.
	 * 
	 * @param dimensions
	 *            the dimensions
	 */
	protected Instances(int dimensions) {
		this.dimensions = dimensions;
		this.instances = new ArrayList<I>();
	}

	/**
	 * Instantiates a new instances.
	 * 
	 * @param dimensions
	 *            the dimensions
	 * @param instances
	 *            the instances
	 */
	protected Instances(int dimensions, ArrayList<I> instances) {
		this(dimensions);
		this.instances = instances;

	}

	/**
	 * Removes the.
	 * 
	 * @param index
	 *            the index
	 */
	protected void remove(int index) {
		I inst = instances.get(index);
		remove(inst);
	}

	/**
	 * Removes the.
	 * 
	 * @param x
	 *            the x
	 * @return true, if successful
	 */
	protected boolean remove(I x) {
		boolean check = instances.remove(x);
		return check;
	}

	/**
	 * Make instances.
	 * 
	 * @param dimensions
	 *            the dimensions
	 * @return the instances
	 */
	protected abstract Instances<I> makeIstances(int dimensions);

	/**
	 * Make instances.
	 * 
	 * @param dimensions
	 *            the dimensions
	 * @param input
	 *            the input
	 * @return the instances
	 */
	protected abstract Instances<I> makeIstances(int dimensions, I... input);

	/**
	 * Make instances.
	 * 
	 * @param dimensions
	 *            the dimensions
	 * @param instances
	 *            the instances
	 * @return the instances
	 */
	protected abstract Instances<I> makeIstances(int dimensions,
			List<I> instances);

	/**
	 * Make instances.
	 * 
	 * @param dimensions
	 *            the dimensions
	 * @param insts
	 *            the insts
	 * @return the instances
	 */
	protected abstract Instances<I> makeIstances(int dimensions,
			Iterator<I> insts);

	/**
	 * Adds the instance.
	 * 
	 * @param instance
	 *            the instance
	 */
	public void addInstance(I instance) {
		this.instances.add(instance);
	}

	/**
	 * Adds the instance.
	 * 
	 * @param instances
	 *            the instances
	 */
	public void addInstance(Collection<I> insts) {
		for (I x : insts) {
			instances.add(x);
		}

	}

	/**
	 * Projects all the instances using projection. 
	 * 
	 * @param <T>
	 *            the generic type
	 * @param projection
	 *            the projection
	 * @return the t
	 */
	@SuppressWarnings("unchecked")
	public <T extends Instances<? extends I>> T project(Projection projection) {
		int outDim = projection.getOutputDimension();
		checkArgument(
				projection.getInputDimension() == this.dimensions,
				"input dimension of projection (%s) and dimension of instances (%s) must match!",
				projection.getInputDimension(), dimensions);

		Instances<I> projected = makeIstances(outDim);
		for (I x : this) {
			projected.addInstance(projection.project(x));
		}
		return (T) projected;
	}

	/**
	 * Splits the instances into two groups based on the value of a particular 
	 * feature given by dimension dim. 
	 * 
	 * @param <T>
	 *            the generic type
	 * @param dim
	 *            the dim
	 * @param value
	 *            the value
	 * @return the generic pair
	 */
	@SuppressWarnings("unchecked")
	public <T extends Instances<? extends I>> GenericPair<T, T> splitOnValue(
			int dim, double value) {
		Instances<I> left = makeIstances(dimensions);
		Instances<I> right = makeIstances(dimensions);

		for (I x : this) {
			if (x.getFeatureValue(dim) <= value) {
				left.addInstance(x);
			} else {
				right.addInstance(x);
			}
		}
		return new GenericPair<T, T>((T) left, (T) right);
	}

	/**
	 * Gets the training instances. You need to specify the fold id and the total 
	 * number of folds. 
	 * 
	 * @param <T>
	 *            the generic type
	 * @param fold
	 *            the fold id
	 * @param numFolds
	 *            the total number of folds
	 * @return the training
	 */
	@SuppressWarnings("unchecked")
	public <T extends Instances<I>> T getTraining(int fold, int numFolds) {
		Instances<I> training = makeIstances(dimensions);
		int foldSize = this.size() / numFolds;
		int start = fold * foldSize;
		int end = (int) Math.min(start + foldSize, this.size());
		for (int i = 0; i < this.size(); i++) {
			if (i < start || i >= end) {
				training.addInstance(this.getInstance(i));
			}
		}
		return (T) training;
	}

	/**
	 * Adds the instance.
	 * 
	 * @param instances
	 *            the instances
	 */
	public void addInstance(Instances<I> instances) {
		for (I x : instances.getInstances()) {
			this.addInstance(x);
		}
	}

	/**
	 * Gets the testing instances. You need to specify the fold id 
	 * and the total number of folds
	 * 
	 * @param <T>
	 *            the generic type
	 * @param fold
	 *            the fold id
	 * @param numFolds
	 *            the total number of folds
	 * @return the testing
	 */
	@SuppressWarnings("unchecked")
	public <T extends Instances<I>> T getTesting(int fold, int numFolds) {
		Instances<I> testing = makeIstances(dimensions);
		int foldSize = this.size() / numFolds;
		int start = fold * foldSize;
		int end = (int) Math.min(start + foldSize, this.size());
		for (int i = 0; i < this.size(); i++) {
			if (i >= start && i < end) {
				testing.addInstance(this.getInstance(i));
			}
		}
		return (T) testing;
	}

	/**
	 * Gets the bag.
	 * 
	 * @param pct
	 *            the pct
	 * @return the bag
	 */
	public Bag<I> getBag(double pct) {
		return getBag((int) (this.size() * pct));
	}

	/**
	 * Gets the bag.
	 * 
	 * @param size
	 *            the size
	 * @return the bag
	 */
	public Bag<I> getBag(int size) {
		if (size > this.size())
			size = this.size();
		Random generator = new Random();
		Instances<I> bag = makeIstances(dimensions);
		Instances<I> out = makeIstances(dimensions);
		Bag<I> bagger = new Bag<I>(bag, out);
		boolean[] check = new boolean[size()];

		for (int i = 0; i < size; i++) {
			int index = generator.nextInt(size);
			bagger.addInBag(this.getInstance(index));
			check[index] = true;
		}
		// out of bag sample
		for (int i = 0; i < check.length; i++) {
			if (!check[i])
				bagger.addOutOfBag(this.getInstance(i));
		}
		return bagger;
	}

	/**
	 * Gets the bag.
	 * 
	 * @return the bag
	 */
	public Bag<I> getBag() {
		return this.getBag(this.size());
	}

	/**
	 * Gets all the instances whose list of indices are supplied as an argument
	 * 
	 * @param indexes
	 *            the indexes
	 * @return the instances
	 */
	public Instances<I> getInstances(List<Integer> indexes) {
		Instances<I> out = makeIstances(dimensions);
		for (int index : indexes) {
			out.addInstance(this.getInstance(index));
		}
		return out;
	}

	/**
	 * Gets the single instance of Instances.
	 * 
	 * @param index
	 *            the index
	 * @return single instance of Instances
	 */
	public I getInstance(int index) {
		return this.instances.get(index);
	}

	/**
	 * Gets a random instance.
	 * 
	 * @return the i
	 */
	public I randomInstance() {
		Random rand = new Random();
		int index = rand.nextInt(size());
		return getInstance(index);
	}

	/**
	 * Gets all the instances. 
	 * 
	 * @return the instances
	 */
	public List<I> getInstances() {
		return this.instances;
	}

	/**
	 * Gets the size- the number of instances in the collection.
	 * 
	 * @return the size
	 */
	public int size() {
		return this.instances.size();
	}

	/**
	 * Gets the dimensions.
	 * 
	 * @return the dimensions
	 */
	public int getDimensions() {
		return this.dimensions;
	}

	/**
	 * Randomly splits the instances into two parts. The size of the first part is 
	 * percent times the total size. The second part consists of all the instances left after 
	 * filling the first part. The two instances objects are packed into a bag and the 
	 * bag is returned.
	 * <p>	  
	 * <b>FIX ME:</b> there should be a check on the value of percent. It should always be
	 * between (0, 1) 
	 * 
	 *   
	 * @param percent
	 *            the percent
	 * @return the bag
	 */
	public Bag<I> randomSplit(double percent) {
		
		@SuppressWarnings("unchecked")
		Instances<I>[] split = new Instances[2];
		Vector<Integer> indexes = new Vector<Integer>();
		for (int i = 0; i < this.size(); i++)
			indexes.add(i);
		Collections.shuffle(indexes);
		double first = ((double) this.size()) * percent;
		split[0] = makeIstances(dimensions);
		split[1] = makeIstances(dimensions);
		for (int i = 0; i < this.size(); i++) {
			if (i < first)
				split[0].addInstance(this.getInstance(indexes.get(i)));
			else
				split[1].addInstance(this.getInstance(indexes.get(i)));
		}
		return new Bag<I>(split[0], split[1]);
	}

	/**
	 * Randomly splits the instances into two parts and returns the first part. The size of the 
	 * first part is equal to percent times the total number of instances 
	 * 
	 * @param percent
	 *            the percent
	 * @return the instances
	 */
	public Instances<I> randomSplitOne(double percent) {
		return randomSplit(percent).getBagInstances();
	}

	/**
	 * Percent split.
	 * 
	 * @param percent
	 *            the percent
	 * @return the bag
	 */
	public Bag<I> percentSplit(double percent) {
		@SuppressWarnings("unchecked")
		Instances<I>[] split = new Instances[2];
		double count = 0.0;
		split[0] = makeIstances(dimensions);
		split[1] = makeIstances(dimensions);
		for (int i = 0; i < this.size(); i++) {
			count += percent;
			if (count >= 1) {
				split[0].addInstance(this.getInstance(i));
				count = 0.0;
			} else {
				split[1].addInstance(this.getInstance(i));
			}
		}
		return new Bag<I>(split[0], split[1]);
	}

	/**
	 * Binary split.
	 * 
	 * @param index
	 *            the index
	 * @return the bag
	 */
	public Bag<I> binarySplit(int index) {
		@SuppressWarnings("unchecked")
		Instances<I>[] split = new Instances[2];
		split[0] = makeIstances(this.dimensions);
		split[1] = makeIstances(this.dimensions);
		for (I x : this.instances) {
			if (x.getFeatureValue(index) != 0) {
				split[1].addInstance(x);
			} else {
				split[0].addInstance(x);
			}
		}
		return new Bag<I>(split[0], split[1]);
	}

	/**
	 * Shuffle.
	 * 
	 * @param <T>
	 *            the generic type
	 * @return the t
	 */
	@SuppressWarnings("unchecked")
	public <T extends Instances<I>> T shuffle() {
		Instances<I> out = makeIstances(this.dimensions);
		for (I inst : this)
			out.addInstance(inst);
		Collections.shuffle(out.instances);

		return (T) out;
	}

	/**
	 * Shallow copy.
	 * 
	 * @return the instances
	 */
	public Instances<I> shallowCopy() {
		Instances<I> out = makeIstances(this.dimensions);
		for (I x : this)
			out.addInstance(x);
		return out;
	}

	/**
	 * Gets the feature vectors.
	 * 
	 * @return the feature vectors
	 */
	public Collection<LinearVector> getFeatureVectors() {
		Collection<LinearVector> out = Lists.newLinkedList();
		for (I x : this)
			out.add(x.getFeatureValues());
		return out;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Iterable#iterator()
	 */
	@Override
	public Iterator<I> iterator() {
		return instances.iterator();
	}

	@Override
	public Object[] toArray() {
		return instances.toArray();
	}

	@Override
	public <T> T[] toArray(T[] arr) {
		return (T[]) instances.toArray(arr);
	}

	@Override
	public boolean retainAll(Collection<?> c) {
		return instances.retainAll(c);
	}

	@Override
	public boolean removeAll(Collection<?> c) {
		return instances.removeAll(c);
	}

	@Override
	public boolean remove(Object o) {
		return instances.remove(o);
	}

	@Override
	public boolean isEmpty() {
		return instances.isEmpty();
	}

	@Override
	public boolean add(I inst) {
		return instances.add(inst);
	}

	@Override
	public boolean addAll(Collection<? extends I> insts) {
		addInstance(Lists.newArrayList(insts));
		return true;
	}

	@Override
	public boolean contains(Object o) {
		return instances.contains(o);
	}

	@Override
	public boolean containsAll(Collection<?> os) {
		return instances.containsAll(os);
	}
}
