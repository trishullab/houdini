
console.log('Executing outside function');
$( document ).ready(function() {
	console.log('Executing container function');
	$( ".Programs" ).wrap(function() {
	  return "<div class='ProgramsContainer'></div>";
	});

	$(".ProgramsContainer").prepend("<div class='TitleBar'><span class='ExpandBtn'><a>Expand</a></span> Programs</div>")


	$(".ExpandBtn").click(function() {
	   $(this).toggleClass('Open');
	   $container = $(this).parent().parent();
	   $container.children('.Programs').slideToggle("slow");
	   
	   if ($(this).hasClass("Open")) {
            $(this).html('<a>Collapse</a>');
       } else{
            $(this).html('<a>Expand</a>');    
       }
	});
});
