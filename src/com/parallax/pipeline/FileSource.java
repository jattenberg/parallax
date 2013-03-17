/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline;

import java.io.File;
import java.lang.reflect.Type;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

import com.google.common.collect.Lists;
import com.google.gson.reflect.TypeToken;

/**
 * FileSource accepts one or more files and generates generic iterator of context
 *
 * @author Josh Attenberg
 */
public class FileSource implements Source<File> {

    private static final long serialVersionUID = 5309685917887826210L;
    private List<File> files;
    private Iterator<Context<File>> it;

    /**
     * Class constructor specifying file path to create
     * @param file file path
     */
    public FileSource(String file) {
        files = Lists.newArrayList(new File(file));
        it = new GenericContextIterator<File>(this.files.iterator());
    }

    /**
     * Class constructor specifying file object to create
     * @param file file object
     */
    public FileSource(File file) {
        files = Lists.newArrayList(file);
        it = new GenericContextIterator<File>(this.files.iterator());
    }

    /**
     * Class constructor specifying multiple files to create
     * @param files multiple files
     */
    public FileSource(Collection<File> files) {
        files = Lists.newArrayList(files);
        it = new GenericContextIterator<File>(this.files.iterator());
    }

    /**
     * Class constructor specifying multiple file path to create
     * @param files multiple file path
     */
    public FileSource(String ... files) {
        this(filesFromStrings(files));
        it = new GenericContextIterator<File>(this.files.iterator());
    }

    /**
     * Class constructor specifying file array to create
     * @param files file array
     */
    public FileSource(File ... files) {
        this.files = Lists.newArrayList(files);
        it = new GenericContextIterator<File>(this.files.iterator());
    }

    private static File[] filesFromStrings(String ... input) {
        File[] out = new File[input.length];
        for(int i = 0; i < input.length; i++)
            out[i] = new File(input[i]);
        return out;
    }
    
    /**
     * The method returns the iterator over context of type File
     * @return Iterator over file context
     */
    @Override
    public Iterator<Context<File>> provideData() {
        return it;
    }

    /**
     * The method returns the class's Type "FoldFilterPipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<FileSource>(){}.getType();
	}
}
