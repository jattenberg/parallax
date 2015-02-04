/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.file;

import com.dsi.parallax.pipeline.AbstractExpandingPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.common.collect.Iterators;
import com.google.gson.reflect.TypeToken;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.IOFileFilter;
import org.apache.commons.io.filefilter.TrueFileFilter;

import java.io.File;
import java.lang.reflect.Type;
import java.util.Iterator;

/**
 * DirectoryToFilesPipe is primitive class for getting contents of File object into context of File
 *
 * @author Josh Attenberg
 */
public class DirectoryToFilesPipe extends AbstractExpandingPipe<File,File> {
  
    private static final long serialVersionUID = 5983509518517454262L;
    private IOFileFilter fileFilter;

    /**
     * Class constructor specifying io file filter to create
     * @param filter io file filter
     */
    public DirectoryToFilesPipe(IOFileFilter filter) {
    	super();
        this.fileFilter = filter;
    }

    /**
     * Class constructor
     */
    public DirectoryToFilesPipe() {
        this(TrueFileFilter.INSTANCE);
    }

    /**
     * Class constructor specifying regular expression string to create
     * @param pattern regular expression
     */
    public DirectoryToFilesPipe(String pattern) {
        this(new RegexFileFilter(pattern));
    }

    /**
     * The method returns the class's Type "DirectoryToFilesPipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<DirectoryToFilesPipe>(){}.getType();
	}

	@Override
	protected Iterator<Context<File>> operate(Context<File> context) {
        File file = context.getData();
        if(file.isDirectory()) {
        	return Iterators.transform(FileUtils.iterateFiles(file, fileFilter, TrueFileFilter.INSTANCE), contextAddingFunction);
        } else {
        	return Iterators.transform(Iterators.forArray(file), contextAddingFunction);
        }
	}
}
