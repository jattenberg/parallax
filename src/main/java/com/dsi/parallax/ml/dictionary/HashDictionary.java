/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.dictionary;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.Collection;
import java.util.Map;

import com.dsi.parallax.ml.util.HashFunctionType;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.google.common.collect.Maps;

/**
 * HashDictionary
 * 
 * @author Josh Attenberg
 */
public class HashDictionary extends AbstractDictionary {
	private final static Map<String, Collection<String>> DUMMYMAP = Maps
			.newHashMap();
	private final static String DUMMYKEY = "key";
	private static final long serialVersionUID = -8231160920621247248L;
	private HashFunctionType hashtype = HashFunctionType.JAVA;
	private boolean biasHash = false;
	private int multipleHashes = 1;

	/**
	 * Class constructor specifying bins to create
	 * 
	 * @param bins
	 *            bins
	 */
	public HashDictionary(int bins) {
		super(bins);
		checkArgument(bins > 0, "bins must be positive: %s", bins);
	}

	/**
	 * Class constructor specifying bins and multiple hashes to create
	 * 
	 * @param bins
	 *            bins
	 * @param multipleHashes
	 *            multiple hashes
	 */
	public HashDictionary(int bins, int multipleHashes) {
		this(bins);
		checkArgument(multipleHashes > 0,
				"multipleHashes must be positive: %s", multipleHashes);
		this.multipleHashes = multipleHashes;
	}

	/**
	 * Class constructor specifying bins and binary features to create
	 * 
	 * @param bins
	 *            bins
	 * @param binaryFeatures
	 *            binary features
	 */
	public HashDictionary(int bins, boolean binaryFeatures) {
		this(bins);
		this.binaryFeatures = binaryFeatures;
	}

	/**
	 * Class constructor specifying bins, binary features and multiple hashes to
	 * create
	 * 
	 * @param bins
	 *            bins
	 * @param binaryFeatures
	 *            binary features
	 * @param multipleHashes
	 *            multiple hashes
	 */
	public HashDictionary(int bins, boolean binaryFeatures, int multipleHashes) {
		this(bins, multipleHashes);
		this.binaryFeatures = binaryFeatures;

	}

	/**
	 * Class constructor specifying bins, binary features and bias hash to
	 * create
	 * 
	 * @param bins
	 *            bins
	 * @param binaryFeatures
	 *            binary features
	 * @param biasHash
	 *            bias hash
	 */
	public HashDictionary(int bins, boolean binaryFeatures, boolean biasHash) {
		this(bins, binaryFeatures);
		this.biasHash = biasHash;
	}

	/**
	 * Class constructor specifying bins, binary features, bias hash and
	 * multiple hashes to create
	 * 
	 * @param bins
	 *            bins
	 * @param binaryFeatures
	 *            binary features
	 * @param biasHash
	 *            bias hash
	 * @param multipleHashes
	 *            multiple hashes
	 */
	public HashDictionary(int bins, boolean binaryFeatures, boolean biasHash,
			int multipleHashes) {
		this(bins, binaryFeatures, multipleHashes);
		this.biasHash = biasHash;
	}

	/**
	 * Class constructor specifying bins and hash type to create
	 * 
	 * @param bins
	 *            bins
	 * @param hashtype
	 *            hash type
	 */
	public HashDictionary(int bins, HashFunctionType hashtype) {
		this(bins);
		this.hashtype = hashtype;
	}

	/**
	 * Class constructor specifying bins, hash type and multiple hashes to
	 * create
	 * 
	 * @param bins
	 *            bins
	 * @param hashtype
	 *            hash type
	 * @param multipleHashes
	 *            multiple hash
	 */
	public HashDictionary(int bins, HashFunctionType hashtype,
			int multipleHashes) {
		this(bins, multipleHashes);
		this.hashtype = hashtype;
	}

	/**
	 * Class constructor specifying bins, hash type and binary features to
	 * create
	 * 
	 * @param bins
	 *            bins
	 * @param binaryFeatures
	 *            binary features
	 * @param hashtype
	 *            hash type
	 */
	public HashDictionary(int bins, boolean binaryFeatures,
			HashFunctionType hashtype) {
		this(bins, binaryFeatures);
		this.hashtype = hashtype;
	}

	/**
	 * Class constructor specifying bins, hash type, binary features and
	 * multiple hashes to create
	 * 
	 * @param bins
	 *            bins
	 * @param binaryFeatures
	 *            binary features
	 * @param hashtype
	 *            hash type
	 * @param multipleHashes
	 *            multiple hashes
	 */
	public HashDictionary(int bins, boolean binaryFeatures,
			HashFunctionType hashtype, int multipleHashes) {
		this(bins, binaryFeatures, multipleHashes);
		this.hashtype = hashtype;
	}

	/**
	 * Class constructor specifying bins, hash type, binary features and bias
	 * hash to create
	 * 
	 * @param bins
	 *            bins
	 * @param binaryFeatures
	 *            binary features
	 * @param biasHash
	 *            bias hash
	 * @param hashtype
	 *            hash type
	 */
	public HashDictionary(int bins, boolean binaryFeatures, boolean biasHash,
			HashFunctionType hashtype) {
		this(bins, binaryFeatures, hashtype);
		this.biasHash = biasHash;
	}

	/**
	 * Class constructor specifying bins, hash type, binary features, bias hash
	 * and multiple hashes to create
	 * 
	 * @param bins
	 *            bins
	 * @param binaryFeatures
	 *            binary features
	 * @param biasHash
	 *            bins hash
	 * @param hashtype
	 *            hash type
	 * @param multipleHashes
	 *            multiple hashes
	 */
	public HashDictionary(int bins, boolean binaryFeatures, boolean biasHash,
			HashFunctionType hashtype, int multipleHashes) {
		this(bins, binaryFeatures, hashtype, multipleHashes);
		this.biasHash = biasHash;
	}

	/**
	 * The method sets bias hash
	 * 
	 * @param biasHash
	 *            bias hash
	 */
	public void setBiasHash(boolean biasHash) {
		this.biasHash = biasHash;
	}

	/**
	 * The method creates linear vector by multiple text
	 * 
	 * @param text
	 *            text
	 * @return LinearVector linear vector
	 */
	@Override
	public LinearVector vectorFromText(Collection<String> text) {
		DUMMYMAP.put(DUMMYKEY, text);
		return vectorFromNamespacedText(DUMMYMAP, false);
	}

	/**
	 * The method creates linear vector by namespace text and name space, of
	 * course, it uses multiple hashes, hash type, binary features and bias hash
	 * 
	 * @param namespacedText
	 *            namespace text
	 * @param namespace
	 *            namespace
	 * @return linear vector
	 */
	@Override
	public LinearVector vectorFromNamespacedText(
			Map<String, Collection<String>> namespacedText, boolean namespace) {
		LinearVector vector = LinearVectorFactory.getVector(dimension);
		for (String ns : namespacedText.keySet()) {
			Collection<String> tokens = namespacedText.get(ns);
			for (String token : tokens) {
				for (int hash = 0; hash < multipleHashes; hash++) {
					String mapToken = makeToken(ns, token, hash, namespace);
					int index = hashtype.hash(mapToken, dimension);
					if (binaryFeatures)
						vector.resetValue(index, 1);
					else {
						int bias = biasHash && hashtype.hash(mapToken, 2) != 1 ? -1
								: 1;
						vector.updateValue(index, bias);
					}
				}
			}
		}
		return vector;
	}

	private String makeToken(String ns, String token, int hash,
			boolean namespace) {
		return (namespace ? ns + "___" : "") + token
				+ (multipleHashes > 1 ? "___" + hash : "");
	}

	@Override
	public int dimensionFromString(String input) {
		return hashtype.hash(input, dimension);
	}
}
