/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.projection;

import java.util.Arrays;

import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;

// TODO: Auto-generated Javadoc
/**
 * does random projection the brute force way (array of doubles)
 * this is an artifact of something old
 * can't be used on high dimensional data
 * TODO: investigate where this is used, maybe delete.
 *
 * @author jattenberg
 */
public class ProjectionMatrix 
{
	
	/** The matrix. */
	float[][] matrix;
	
	/** The in. */
	int in;
	
	/** The out. */
	int out;
	
	/**
	 * Instantiates a new projection matrix.
	 *
	 * @param in the in
	 * @param out the out
	 */
	public ProjectionMatrix(int in, int out)
	{
		this.in = in;
		this.out = out;
		matrix = new float[out][in];
		initialize();
	}	
	
	/**
	 * Initialize.
	 */
	private void initialize()
	{
		for(int i = 0; i < out; i ++)
		{
			for(int j = 0; j < in; j++)
			{
				matrix[i][j] = (float)(MLUtils.GENERATOR.nextFloat()*2.0 - 1.0);
			}
		}
		gramSchmidt();
	}
	
	/**
	 * Proj vonto u.
	 *
	 * @param v the v
	 * @param u the u
	 * @return the float[]
	 */
	private float[] projVontoU(int v, int u)
	{
		float[] x = new float[in];

		float ip = innerProduct(matrix[v],matrix[u]);
		float norm = rowNorm(u);
		if(norm != 0)
		{
			float mult = ip/norm;
			for(int i = 0; i < in; i++)
				x[i] = mult*matrix[u][i];
		}
		return x;
	}
	
	/**
	 * Inner product.
	 *
	 * @param v the v
	 * @param u the u
	 * @return the float
	 */
	private float innerProduct(float[] v, float[] u)
	{
		float ip = 0.0f;
		for(int i = 0; i < v.length; i++)
			ip += (v[i] - u[i])*(v[i] - u[i]);
		return (float)Math.sqrt(ip);
	}
	
	/**
	 * Gram schmidt.
	 */
	private void gramSchmidt()
	{

		for(int i = 0; i < out; i++)
		{
			float[] ti = Arrays.copyOf(matrix[i], in);
			for(int j = 0; j < i; j++)
			{
				float[] proj = projVontoU(i,j);
				for(int k = 0; k < in; k++)
					ti[k] -= proj[k];
			}
			float n = norm(ti);
			if(n != 0)
				for(int k = 0; k < in; k++)
					ti[k] /= n;
			matrix[i] = Arrays.copyOf(ti, in);
		}
	}
	
	/**
	 * Project.
	 *
	 * @param x the x
	 * @return the instanze
	 */
	public Instance<?> project(Instance<?> x)
	{
		LinearVector vec = LinearVectorFactory.getDenseVector(out);
		for(int x_i : x)
		{
			double y_i = x.getFeatureValue(x_i);
			for(int i = 0; i < out; i++)
			{
				vec.updateValue(i, matrix[i][x_i]*y_i);
			}
		}
		return x.cloneNewVector(vec);
	}
	
	/**
	 * Row norm.
	 *
	 * @param row the row
	 * @return the float
	 */
	private float rowNorm(int row)
	{
		return norm(matrix[row]);
	}
	
	/**
	 * Norm.
	 *
	 * @param x the x
	 * @return the float
	 */
	private float norm(float[] x)
	{
		float norm = 0.0f;
		for(int i = 0; i < x.length; i++)
			norm += x[i]*x[i];
		return (float)Math.sqrt(norm);
	}
	
	/* (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public String toString()
	{
		StringBuffer buff = new StringBuffer(Arrays.toString(matrix[0]));
		for(int i = 1; i < out; i++)
			buff.append("\n"+Arrays.toString(matrix[i]));
		
		
		return buff.toString();
	}
	
	/**
	 * The main method.
	 *
	 * @param args the arguments
	 * @throws Exception the exception
	 */
	public static void main(String[] args) throws Exception
	{
		ProjectionMatrix p = new ProjectionMatrix(100, 5);
		System.out.println(p.toString());
		
	}
}
