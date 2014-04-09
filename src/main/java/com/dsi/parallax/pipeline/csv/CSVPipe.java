package com.dsi.parallax.pipeline.csv;

import java.lang.reflect.Type;
import java.util.List;

import com.dsi.parallax.ml.util.csv.CSV;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.pipeline.AbstractPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.gson.reflect.TypeToken;

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
