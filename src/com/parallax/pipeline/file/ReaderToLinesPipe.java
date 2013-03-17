package com.parallax.pipeline.file;

import java.io.BufferedReader;
import java.lang.reflect.Type;
import java.util.Iterator;

import com.google.common.collect.Iterators;
import com.google.gson.reflect.TypeToken;
import com.parallax.pipeline.AbstractExpandingPipe;
import com.parallax.pipeline.Context;

public class ReaderToLinesPipe extends
		AbstractExpandingPipe<BufferedReader, String> {

	private static final long serialVersionUID = 575612555124676746L;

	public ReaderToLinesPipe() {
		super();
	}

	@Override
	public Type getType() {
		return new TypeToken<ReaderToLinesPipe>() {
		}.getType();
	}

	@Override
	protected Iterator<Context<String>> operate(Context<BufferedReader> context) {

		return Iterators.transform(
				new BufferedReaderIterable(context.getData()).iterator(),
				contextAddingFunction);

	}

}
