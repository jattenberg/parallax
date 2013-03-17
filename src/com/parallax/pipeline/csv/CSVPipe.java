package com.parallax.pipeline.csv;

import java.lang.reflect.Type;
import java.util.List;

import com.google.gson.reflect.TypeToken;
import com.parallax.ml.util.csv.CSV;
import com.parallax.ml.vector.LinearVector;
import com.parallax.pipeline.AbstractPipe;
import com.parallax.pipeline.Context;

public class CSVPipe extends AbstractPipe<List<String>, LinearVector> {

	private static final long serialVersionUID = -6467743863025738690L;
	private final CSV csv;

	public CSVPipe(CSV csv) {
		super();
		this.csv = csv;
	}

	@Override
	public Type getType() {
		return new TypeToken<CSVPipe>() {
		}.getType();
	}

	@Override
	protected Context<LinearVector> operate(Context<List<String>> context) {
		return csv.parseRow(context.getData());
	}

}
