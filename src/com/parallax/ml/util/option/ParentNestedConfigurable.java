package com.parallax.ml.util.option;

/**
 * The Class ParentNestedConfigurable- the base class for types that are
 * configurable, having a variety of child configurables, each with their own
 * configurable options.
 * 
 * @param <T>
 *            the generic type- the type for the internal nested configuration-
 *            each nested configurable should correspond to a single enum type
 * @param <C>
 *            the generic type- the base class for the nested configurable
 * @param <P>
 *            the generic type- the concrete class for the internal nested
 *            configurable
 */
public abstract class ParentNestedConfigurable<T extends Enum<T> & NestedConfigurableType<C, T>, C extends NestedConfigurable<C, T>, P extends ParentNestedConfigurable<T, C, P>>
		extends AbstractConfigurable<P> {

	/**
	 * in order to share values with sub classes
	 */
	public static final String TYPESHORT = ParentNestedConfigurableOptionSet.TYPESHORT;
	public static final String TYPELONG = ParentNestedConfigurableOptionSet.TYPELONG;

	public static final String CONFIGSHORT = ParentNestedConfigurableOptionSet.CONFIGSHORT;
	public static final String CONFIGLONG = ParentNestedConfigurableOptionSet.CONFIGLONG;

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 5112338372120942090L;

	/** The current type. */
	protected T currentType;

	/** The current configurable. */
	protected C currentConfigurable;

	/** the parent configurable itself, used for method chaining only */
	protected final P parentConfigurable;
	
	
	protected ParentNestedConfigurable() {
		this.parentConfigurable = getParentConfigurable();
	}
	
	/**
	 * get the parent nested configurable
	 * all concrete implementations should implement and return [this]
	 * @return
	 */
	protected abstract P getParentConfigurable();
	
	/**
	 * Child types.
	 * 
	 * @return the t[]
	 */
	public abstract T[] childTypes();

	/**
	 * Configurable for type.
	 * 
	 * @param type
	 *            the type
	 * @return the c
	 */
	public abstract C configurableForType(T type);

	/**
	 * Current nested configuration.
	 * 
	 * @return the configuration
	 */
	public Configuration<C> currentNestedConfiguration() {
		return currentConfigurable.getConfiguration();
	}

	/**
	 * Current nested configurable.
	 * 
	 * @return the c
	 */
	public abstract C currentNestedConfigurable();

	/**
	 * Gets the current type.
	 * 
	 * @return the current type
	 */
	public T getCurrentType() {
		return currentType;
	}

	
	public P setConfigurableType(T type) {
		this.currentType = type;
		this.currentConfigurable = type.getConfigurable();
		return parentConfigurable;
	}
}
