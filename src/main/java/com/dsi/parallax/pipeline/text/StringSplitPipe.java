package com.dsi.parallax.pipeline.text;

import java.io.IOException;
import java.lang.reflect.Type;
import java.util.Arrays;
import java.util.List;

import org.apache.log4j.Logger;

import au.com.bytecode.opencsv.CSVParser;

import com.dsi.parallax.pipeline.AbstractPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.gson.reflect.TypeToken;

public class StringSplitPipe extends AbstractPipe<String, List<String>> {

	private static final Logger LOGGER = Logger
			.getLogger(StringSplitPipe.class);

	private static final long serialVersionUID = -6055008751257883317L;
	private final static char DEFAULT_DELIM = ',';
	private final CSVParser parser;

	public StringSplitPipe() {
		this(DEFAULT_DELIM);
	}

	public StringSplitPipe(char delim) {
		super();
		parser = new CSVParser(delim);
	}

	@Override
	public Type getType() {
		return new TypeToken<StringSplitPipe>() {
		}.getType();
	}

	@Override
	protected Context<List<String>> operate(Context<String> context) {
		try {
			return Context.createContext(context,
					Arrays.asList(parser.parseLine(context.getData())));
		} catch (IOException e) {
			LOGGER.error(e.getLocalizedMessage());
			e.printStackTrace();
		}
		// masks the error :(
		return null;
	}

}
