DIRS = common opengl cuda

.PHONY: debug

libs:
	@for d in $(DIRS); do echo "making" $$d ; $(MAKE) $(MAKECMDGOALS) -C $$d ; done

debug: libs

clean:
	@for d in $(DIRS); do echo "cleaning" $$d ; $(MAKE) clean -C $$d ; done
