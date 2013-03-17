/*******************************************************************************
 * Copyright 2012 Josh Att
import com.parallax.ml.vector.LinearVectorFactory;

import com.parallax.ml.vector.LinearVectorFactory;

import com.parallax.ml.vector.LinearVector;
enberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import com.google.common.collect.Lists;
import com.parallax.ml.vector.LinearVector;
import com.parallax.ml.vector.LinearVectorFactory;
import static com.google.common.base.Preconditions.checkArgument;

public abstract class KDTree<T> {
	// Static variables
	private static final int bucketSize = 24;

	// All types
	private final int dimensions;
	private final KDTree<T> parent;

	// Root only
	private final LinkedList<LinearVector> locationStack;
	private final Integer sizeLimit;

	// Leaf only
	private LinearVector[] locations;
	private Object[] data;
	private int locationCount;

	// Stem only
	private KDTree<T> left, right;
	private int splitDimension;
	private double splitValue;

	// Bounds
	private LinearVector minLimit, maxLimit;
	private boolean singularity;

	// Temporary
	private Status status;

	/**
	 * Construct a KDTree with a given number of dimensions and a limit on
	 * maxiumum size (after which it throws away old points)
	 */
	public KDTree(int dimensions, Integer sizeLimit) {
		this.dimensions = dimensions;

		// Init as leaf
		this.locations = new LinearVector[bucketSize];
		this.data = new Object[bucketSize];
		this.locationCount = 0;
		this.singularity = true;

		// Init as root
		this.parent = null;
		this.sizeLimit = sizeLimit;
		if (sizeLimit != null) {
			this.locationStack = Lists.newLinkedList();
		} else {
			this.locationStack = null;
		}
	}

	/**
	 * Constructor for child nodes. Internal use only.
	 */
	private KDTree(KDTree<T> parent, boolean right) {
		this.dimensions = parent.dimensions;

		// Init as leaf
		this.locations = new LinearVector[Math.max(bucketSize,
				parent.locationCount)];
		this.data = new Object[Math.max(bucketSize, parent.locationCount)];
		this.locationCount = 0;
		this.singularity = true;

		// Init as non-root
		this.parent = parent;
		this.locationStack = null;
		this.sizeLimit = null;
	}

	/**
	 * Get the number of points in the tree
	 */
	public int size() {
		return locationCount;
	}

	/**
	 * Add a point and associated value to the tree
	 */
	public void addPoint(LinearVector location, T value) {
		checkArgument(location.size() == dimensions,
				"input location size (%s) must equal kd-tree dimension (%s)",
				location.size(), dimensions);
		KDTree<T> cursor = this;

		while (cursor.locations == null
				|| cursor.locationCount >= cursor.locations.length) {
			if (cursor.locations != null) {
				cursor.splitDimension = cursor.findWidestAxis();
				cursor.splitValue = (cursor.minLimit
						.getValue(cursor.splitDimension) + cursor.maxLimit
						.getValue(cursor.splitDimension)) * 0.5;

				// Never split on infinity or NaN
				if (cursor.splitValue == Double.POSITIVE_INFINITY) {
					cursor.splitValue = Double.MAX_VALUE;
				} else if (cursor.splitValue == Double.NEGATIVE_INFINITY) {
					cursor.splitValue = -Double.MAX_VALUE;
				} else if (Double.isNaN(cursor.splitValue)) {
					cursor.splitValue = 0;
				}

				// Don't split node if it has no width in any axis. Double the
				// bucket size instead
				if (cursor.minLimit.getValue(cursor.splitDimension) == cursor.maxLimit
						.getValue(cursor.splitDimension)) {
					LinearVector[] newLocations = new LinearVector[cursor.locations.length * 2];
					System.arraycopy(cursor.locations, 0, newLocations, 0,
							cursor.locationCount);
					cursor.locations = newLocations;
					Object[] newData = new Object[newLocations.length];
					System.arraycopy(cursor.data, 0, newData, 0,
							cursor.locationCount);
					cursor.data = newData;
					break;
				}

				// Don't let the split value be the same as the upper value as
				// can happen due to rounding errors!
				if (cursor.splitValue == cursor.maxLimit
						.getValue(cursor.splitDimension)) {
					cursor.splitValue = cursor.minLimit
							.getValue(cursor.splitDimension);
				}

				// Create child leaves
				KDTree<T> left = new ChildNode(cursor, false);
				KDTree<T> right = new ChildNode(cursor, true);

				// Move locations into children
				for (int i = 0; i < cursor.locationCount; i++) {
					LinearVector oldLocation = cursor.locations[i];
					Object oldData = cursor.data[i];
					if (oldLocation.getValue(cursor.splitDimension) > cursor.splitValue) {
						// Right
						right.locations[right.locationCount] = oldLocation;
						right.data[right.locationCount] = oldData;
						right.locationCount++;
						right.extendBounds(oldLocation);
					} else {
						// Left
						left.locations[left.locationCount] = oldLocation;
						left.data[left.locationCount] = oldData;
						left.locationCount++;
						left.extendBounds(oldLocation);
					}
				}

				// Make into stem
				cursor.left = left;
				cursor.right = right;
				cursor.locations = null;
				cursor.data = null;
			}

			cursor.locationCount++;
			cursor.extendBounds(location);

			if (location.getValue(cursor.splitDimension) > cursor.splitValue) {
				cursor = cursor.right;
			} else {
				cursor = cursor.left;
			}
		}

		cursor.locations[cursor.locationCount] = location;
		cursor.data[cursor.locationCount] = value;
		cursor.locationCount++;
		cursor.extendBounds(location);

		if (this.sizeLimit != null) {
			this.locationStack.add(location);
			if (this.locationCount > this.sizeLimit) {
				this.removeOld();
			}
		}
	}

	/**
	 * Extends the bounds of this node do include a new location
	 */
	private final void extendBounds(LinearVector location) {
		if (minLimit == null) {
			minLimit = LinearVectorFactory.getVector(location);
			maxLimit = LinearVectorFactory.getVector(location);
			return;
		}

		for (int i = 0; i < dimensions; i++) {
			if (Double.isNaN(location.getValue(i))) {
				minLimit.resetValue(i, Double.NaN);
				maxLimit.resetValue(i, Double.NaN);
				singularity = false;
			} else if (minLimit.getValue(i) > location.getValue(i)) {
				minLimit.resetValue(i, location.getValue(i));
				singularity = false;
			} else if (maxLimit.getValue(i) < location.getValue(i)) {
				maxLimit.resetValue(i, location.getValue(i));
				singularity = false;
			}
		}
	}

	/**
	 * Find the widest axis of the bounds of this node
	 */
	private final int findWidestAxis() {
		int widest = 0;
		double width = (maxLimit.getValue(0) - minLimit.getValue(0))
				* getAxisWeightHint(0);
		if (Double.isNaN(width))
			width = 0;
		for (int i = 1; i < dimensions; i++) {
			double nwidth = (maxLimit.getValue(i) - minLimit.getValue(i))
					* getAxisWeightHint(i);
			if (Double.isNaN(nwidth))
				nwidth = 0;
			if (nwidth > width) {
				widest = i;
				width = nwidth;
			}
		}
		return widest;
	}

	/**
	 * Remove the oldest value from the tree. Note: This cannot trim the bounds
	 * of nodes, nor empty nodes, and thus you can't expect it to perfectly
	 * preserve the speed of the tree as you keep adding.
	 */
	private void removeOld() {
		LinearVector location = this.locationStack.removeFirst();
		KDTree<T> cursor = this;

		// Find the node where the point is
		while (cursor.locations == null) {
			if (location.getValue(cursor.splitDimension) > cursor.splitValue) {
				cursor = cursor.right;
			} else {
				cursor = cursor.left;
			}
		}

		for (int i = 0; i < cursor.locationCount; i++) {
			if (cursor.locations[i] == location) {
				System.arraycopy(cursor.locations, i + 1, cursor.locations, i,
						cursor.locationCount - i - 1);
				cursor.locations[cursor.locationCount - 1] = null;
				System.arraycopy(cursor.data, i + 1, cursor.data, i,
						cursor.locationCount - i - 1);
				cursor.data[cursor.locationCount - 1] = null;
				do {
					cursor.locationCount--;
					cursor = cursor.parent;
				} while (cursor.parent != null);
				return;
			}
		}
		// If we got here... we couldn't find the value to remove. Weird...
	}

	/**
	 * Enumeration representing the status of a node during the running
	 */
	private static enum Status {
		NONE, LEFTVISITED, RIGHTVISITED, ALLVISITED
	}

	/**
	 * Stores a distance and value to output
	 */
	public static class Entry<T> {
		public final double distance;
		public final T value;
		public final LinearVector position;

		private Entry(double distance, T value, LinearVector position) {
			this.distance = distance;
			this.value = value;
			this.position = position;
		}
	}

	/**
	 * Calculates the nearest 'count' points to 'location'
	 */
	@SuppressWarnings("unchecked")
	public List<Entry<T>> nearestNeighbor(LinearVector location, int count,
			boolean sequentialSorting) {
		KDTree<T> cursor = this;
		cursor.status = Status.NONE;
		double range = Double.POSITIVE_INFINITY;
		ResultHeap resultHeap = new ResultHeap(count);

		do {
			if (cursor.status == Status.ALLVISITED) {
				// At a fully visited part. Move up the tree
				cursor = cursor.parent;
				continue;
			}

			if (cursor.status == Status.NONE && cursor.locations != null) {
				// At a leaf. Use the data.
				if (cursor.locationCount > 0) {
					if (cursor.singularity) {
						double dist = pointDist(cursor.locations[0], location);
						if (dist <= range) {
							for (int i = 0; i < cursor.locationCount; i++) {
								resultHeap.addValue(dist, cursor.data[i],
										cursor.locations[i]);
							}
						}
					} else {
						for (int i = 0; i < cursor.locationCount; i++) {
							double dist = pointDist(cursor.locations[i],
									location);
							resultHeap.addValue(dist, cursor.data[i],
									cursor.locations[i]);
						}
					}
					range = resultHeap.getMaxDist();
				}

				if (cursor.parent == null) {
					break;
				}
				cursor = cursor.parent;
				continue;
			}

			// Going to descend
			KDTree<T> nextCursor = null;
			if (cursor.status == Status.NONE) {
				// At a fresh node, descend the most probably useful direction
				if (location.getValue(cursor.splitDimension) > cursor.splitValue) {
					// Descend right
					nextCursor = cursor.right;
					cursor.status = Status.RIGHTVISITED;
				} else {
					// Descend left;
					nextCursor = cursor.left;
					cursor.status = Status.LEFTVISITED;
				}
			} else if (cursor.status == Status.LEFTVISITED) {
				// Left node visited, descend right.
				nextCursor = cursor.right;
				cursor.status = Status.ALLVISITED;
			} else if (cursor.status == Status.RIGHTVISITED) {
				// Right node visited, descend left.
				nextCursor = cursor.left;
				cursor.status = Status.ALLVISITED;
			}

			// Check if it's worth descending. Assume it is if it's sibling has
			// not been visited yet.
			if (cursor.status == Status.ALLVISITED) {
				if (nextCursor.locationCount == 0
						|| (!nextCursor.singularity && pointRegionDist(
								location, nextCursor.minLimit,
								nextCursor.maxLimit) > range)) {
					continue;
				}
			}

			// Descend down the tree
			cursor = nextCursor;
			cursor.status = Status.NONE;
		} while (cursor.parent != null || cursor.status != Status.ALLVISITED);

		ArrayList<Entry<T>> results = new ArrayList<Entry<T>>(resultHeap.values);
		if (sequentialSorting) {
			while (resultHeap.values > 0) {
				resultHeap.removeLargest();
				results.add(new Entry<T>(resultHeap.removedDist,
						(T) resultHeap.removedData, resultHeap.removedPosition));
			}
		} else {
			for (int i = 0; i < resultHeap.values; i++) {
				results.add(new Entry<T>(resultHeap.distance[i],
						(T) resultHeap.data[i], resultHeap.positions[i]));
			}
		}

		return results;
	}

	// Override in subclasses
	protected abstract double pointDist(LinearVector p1, LinearVector p2);

	protected abstract double pointRegionDist(LinearVector point,
			LinearVector min, LinearVector max);

	protected double getAxisWeightHint(int i) {
		return 1.0;
	}

	/**
	 * Internal class for child nodes
	 */
	private class ChildNode extends KDTree<T> {
		private ChildNode(KDTree<T> parent, boolean right) {
			super(parent, right);
		}

		// Distance measurements are always called from the root node
		@Override
		protected double pointDist(LinearVector p1, LinearVector p2) {
			throw new IllegalStateException();
		}

		@Override
		protected double pointRegionDist(LinearVector point, LinearVector min,
				LinearVector max) {
			throw new IllegalStateException();
		}
	}

	/**
	 * Class for tree with Weighted Squared Euclidean distancing
	 */
	public static class WeightedSqrEuclid<T> extends KDTree<T> {
		private double[] weights;

		public WeightedSqrEuclid(int dimensions, Integer sizeLimit) {
			super(dimensions, sizeLimit);
			this.weights = new double[dimensions];
			Arrays.fill(this.weights, 1.0);
		}

		public void setWeights(double[] weights) {
			this.weights = weights;
		}

		protected double getAxisWeightHint(int i) {
			return weights[i];
		}

		@Override
		protected double pointDist(LinearVector p1, LinearVector p2) {
			double d = 0;

			for (int i = 0; i < p1.size(); i++) {
				double diff = (p1.getValue(i) - p2.getValue(i)) * weights[i];
				if (!Double.isNaN(diff)) {
					d += diff * diff;
				}
			}

			return d;
		}

		@Override
		protected double pointRegionDist(LinearVector point, LinearVector min,
				LinearVector max) {
			double d = 0;

			for (int i = 0; i < point.size(); i++) {
				double diff = 0;
				if (point.getValue(i) > max.getValue(i)) {
					diff = (point.getValue(i) - max.getValue(i)) * weights[i];
				} else if (point.getValue(i) < min.getValue(i)) {
					diff = (point.getValue(i) - min.getValue(i)) * weights[i];
				}

				if (!Double.isNaN(diff)) {
					d += diff * diff;
				}
			}

			return d;
		}
	}

	/**
	 * Class for tree with Unweighted Squared Euclidean distancing
	 */
	public static class SqrEuclid<T> extends KDTree<T> {
		public SqrEuclid(int dimensions, Integer sizeLimit) {
			super(dimensions, sizeLimit);
		}

		@Override
		protected double pointDist(LinearVector p1, LinearVector p2) {
			double d = 0;

			for (int i = 0; i < p1.size(); i++) {
				double diff = (p1.getValue(i) - p2.getValue(i));
				if (!Double.isNaN(diff)) {
					d += diff * diff;
				}
			}

			return d;
		}

		protected double pointRegionDist(LinearVector point, LinearVector min,
				LinearVector max) {
			double d = 0;

			for (int i = 0; i < point.size(); i++) {
				double diff = 0;
				if (point.getValue(i) > max.getValue(i)) {
					diff = (point.getValue(i) - max.getValue(i));
				} else if (point.getValue(i) < min.getValue(i)) {
					diff = (point.getValue(i) - min.getValue(i));
				}

				if (!Double.isNaN(diff)) {
					d += diff * diff;
				}
			}

			return d;
		}
	}

	/**
	 * Class for tree with Weighted Manhattan distancing
	 */
	public static class WeightedManhattan<T> extends KDTree<T> {
		private double[] weights;

		public WeightedManhattan(int dimensions, Integer sizeLimit) {
			super(dimensions, sizeLimit);
			this.weights = new double[dimensions];
			Arrays.fill(this.weights, 1.0);
		}

		public void setWeights(double[] weights) {
			this.weights = weights;
		}

		protected double getAxisWeightHint(int i) {
			return weights[i];
		}

		@Override
		protected double pointDist(LinearVector p1, LinearVector p2) {
			double d = 0;

			for (int i = 0; i < p1.size(); i++) {
				double diff = (p1.getValue(i) - p2.getValue(i));
				if (!Double.isNaN(diff)) {
					d += ((diff < 0) ? -diff : diff) * weights[i];
				}
			}

			return d;
		}

		protected double pointRegionDist(LinearVector point, LinearVector min,
				LinearVector max) {
			double d = 0;

			for (int i = 0; i < point.size(); i++) {
				double diff = 0;
				if (point.getValue(i) > max.getValue(i)) {
					diff = (point.getValue(i) - max.getValue(i));
				} else if (point.getValue(i) < min.getValue(i)) {
					diff = (min.getValue(i) - point.getValue(i));
				}

				if (!Double.isNaN(diff)) {
					d += diff * weights[i];
				}
			}

			return d;
		}
	}

	/**
	 * Class for tree with Manhattan distancing
	 */
	public static class Manhattan<T> extends KDTree<T> {
		public Manhattan(int dimensions, Integer sizeLimit) {
			super(dimensions, sizeLimit);
		}

		protected double pointDist(LinearVector p1, LinearVector p2) {
			double d = 0;

			for (int i = 0; i < p1.size(); i++) {
				double diff = (p1.getValue(i) - p2.getValue(i));
				if (!Double.isNaN(diff)) {
					d += (diff < 0) ? -diff : diff;
				}
			}

			return d;
		}

		protected double pointRegionDist(LinearVector point, LinearVector min,
				LinearVector max) {
			double d = 0;

			for (int i = 0; i < point.size(); i++) {
				double diff = 0;
				if (point.getValue(i) > max.getValue(i)) {
					diff = (point.getValue(i) - max.getValue(i));
				} else if (point.getValue(i) < min.getValue(i)) {
					diff = (min.getValue(i) - point.getValue(i));
				}

				if (!Double.isNaN(diff)) {
					d += diff;
				}
			}

			return d;
		}
	}

	/**
	 * Class for tracking up to 'size' closest values
	 */
	private static class ResultHeap {
		private final Object[] data;
		private final double[] distance;
		private final LinearVector[] positions;
		private final int size;
		private int values;
		public Object removedData;
		public double removedDist;
		public LinearVector removedPosition;

		public ResultHeap(int size) {
			this.data = new Object[size];
			this.distance = new double[size];
			this.positions = new LinearVector[size];
			this.size = size;
			this.values = 0;
		}

		public void addValue(double dist, Object value, LinearVector position) {
			// If there is still room in the heap
			if (values < size) {
				// Insert new value at the end
				data[values] = value;
				distance[values] = dist;
				positions[values] = position;
				upHeapify(values);
				values++;
			}
			// If there is no room left in the heap, and the new entry is lower
			// than the max entry
			else if (dist < distance[0]) {
				// Replace the max entry with the new entry
				data[0] = value;
				distance[0] = dist;
				downHeapify(0);
			}
		}

		public void removeLargest() {
			if (values == 0) {
				throw new IllegalStateException();
			}

			removedData = data[0];
			removedDist = distance[0];
			removedPosition = positions[0];
			values--;
			data[0] = data[values];
			distance[0] = distance[values];
			positions[0] = positions[values];
			downHeapify(0);
		}

		private void upHeapify(int c) {
			for (int p = (c - 1) / 2; c != 0 && distance[c] > distance[p]; c = p, p = (c - 1) / 2) {
				Object pData = data[p];
				double pDist = distance[p];
				LinearVector pPosition = positions[p];
				data[p] = data[c];
				distance[p] = distance[c];
				positions[p] = positions[c];

				data[c] = pData;
				distance[c] = pDist;
				positions[c] = pPosition;
			}
		}

		private void downHeapify(int p) {
			for (int c = p * 2 + 1; c < values; p = c, c = p * 2 + 1) {
				if (c + 1 < values && distance[c] < distance[c + 1]) {
					c++;
				}
				if (distance[p] < distance[c]) {
					// Swap the points
					Object pData = data[p];
					double pDist = distance[p];
					LinearVector pPosition = positions[p];

					data[p] = data[c];
					distance[p] = distance[c];
					positions[p] = positions[c];

					data[c] = pData;
					distance[c] = pDist;
					positions[c] = pPosition;
				} else {
					break;
				}
			}
		}

		public double getMaxDist() {
			if (values < size) {
				return Double.POSITIVE_INFINITY;
			}
			return distance[0];
		}
	}
}
